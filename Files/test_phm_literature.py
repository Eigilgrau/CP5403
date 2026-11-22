"""Check source fidelity and boundary rejection with explicitly synthetic fixtures."""

from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import phm_literature as lit


class LiteratureChecks(unittest.TestCase):
    def test_searchable_scan_background_preserves_text_only_when_declared(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "searchable_scan.pdf"
            doc = pymupdf.open()
            page = doc.new_page(width=300, height=400)
            pixel = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), False)
            pixel.clear_with(255)
            page.insert_image(page.rect, stream=pixel.tobytes("png"))
            page.insert_text((30, 120), "The fan operates during occupied hours.")
            doc.save(path)
            doc.close()
            regular = lit.extract_pdf(path, {"layout": "single_column"})
            scanned = lit.extract_pdf(path, {"layout": "single_column", "searchable_scan": True})
            self.assertEqual(len(regular["paragraphs"]), 0)
            self.assertEqual(len(scanned["paragraphs"]), 1)
            self.assertIn("fan operates", scanned["paragraphs"][0]["text"])

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.output = cls.root / "Datasets/PHM15-literature"
        cls.tokenizer = lit.EmbeddingTokenizer(cls.output / "model_assets")
        cls.paragraph = {"paragraph_id": "fixture.p001.b001", "document_id": "fixture", "page": 1,
                         "section": "Fixture", "bbox": [0, 0, 100, 100], "flags": [],
                         "raw_text": "The motor runs. The current rises. The alarm follows.",
                         "text": "The motor runs. The current rises. The alarm follows."}
        cls.units = lit.make_units([cls.paragraph], cls.tokenizer)

    def test_valid_boundaries_preserve_every_unit(self):
        ends = [self.units[1]["unit_id"], self.units[-1]["unit_id"]]
        groups = lit.validate_boundaries({"chunk_end_ids": ends}, self.units, self.tokenizer)
        self.assertEqual([u for g in groups for u in g], self.units)

    def test_malformed_boundaries_are_rejected(self):
        ids = [u["unit_id"] for u in self.units]
        failures = ["not JSON", '```json\n{}\n```', [], {}, {"chunk_end_ids": []},
                    {"chunk_end_ids": [9]}, {"chunk_end_ids": ["unknown"]},
                    {"chunk_end_ids": [ids[0]]}, {"chunk_end_ids": [ids[0], ids[0], ids[-1]]},
                    {"chunk_end_ids": [ids[-1], ids[0]]},
                    {"chunk_end_ids": [ids[-1]], "rewritten_text": "invented"}]
        for response in failures:
            with self.subTest(response=response), self.assertRaises(ValueError):
                lit.validate_boundaries(response, self.units, self.tokenizer)

    def test_long_sentence_is_fragmented_without_loss(self):
        p = deepcopy(self.paragraph)
        p["raw_text"] = p["text"] = " ".join(["measurement"] * 700) + "."
        units = lit.make_units([p], self.tokenizer)
        self.assertGreater(len(units), 1)
        self.assertEqual(lit.joined(units), p["text"])
        self.assertTrue(all(u["fragmented_sentence"] for u in units))
        with self.assertRaisesRegex(ValueError, "embedding_limit"):
            lit.validate_boundaries({"chunk_end_ids": [units[-1]["unit_id"]]}, units, self.tokenizer)

    def test_column_order_uses_layout_not_pdf_object_order(self):
        left = {"bbox": [10, 80, 45, 120], "id": "left"}
        right = {"bbox": [55, 50, 90, 120], "id": "right"}
        title = {"bbox": [10, 10, 90, 20], "id": "title"}
        result = lit.reading_order([right, left, title], 100, "two_columns")
        self.assertEqual([b["id"] for b in result], ["title", "left", "right"])

    def test_fabricated_chunk_text_fails_source_reconstruction(self):
        inventory = {"document_id": "fixture", "source_sha256": "fixture", "filename": "fixture.pdf", "policy": {}}
        chunk = lit.chunk_record(self.units, "synthetic_fixture", self.tokenizer, inventory)
        lit.validate_chunks([chunk], self.units, [self.paragraph], self.tokenizer)
        chunk["text"] += " Invented statement."
        with self.assertRaisesRegex(ValueError, "exact reconstruction"):
            lit.validate_chunks([chunk], self.units, [self.paragraph], self.tokenizer)

    def fixture_paragraphs(self, texts, section="Fixture"):
        result = []
        for i, text in enumerate(texts, 1):
            paragraph = deepcopy(self.paragraph)
            paragraph.update(paragraph_id=f"fixture.p001.b{i:03}", raw_text=text, text=text, section=section)
            result.append(paragraph)
        return result

    def test_context_join_reconstructs_a_sentence_across_blocks(self):
        paragraphs = self.fixture_paragraphs(["The framework combines a language model (LLM)",
                                               "to retrieve source evidence."])
        units = lit.make_units(paragraphs, self.tokenizer)
        groups = [[u] for u in units]
        inventory = {"document_id": "fixture", "source_sha256": "fixture", "filename": "fixture.pdf", "policy": {}}
        records, audit = lit.context_chunk_records(groups, "qwen_boundaries", self.tokenizer, inventory)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], " ".join(p["text"] for p in paragraphs))
        self.assertEqual(records[0]["preparation_method"], "qwen_boundaries_context_join")
        self.assertEqual(records[0]["boundary_selection_method"], "qwen_boundaries")
        self.assertEqual(records[0]["boundary_cleanup"]["original_chunk_end_ids"], [u["unit_id"] for u in units])
        self.assertEqual(audit["joined_boundaries"], 1)
        lit.validate_chunks(records, units, paragraphs, self.tokenizer)

    def test_context_join_keeps_heading_and_list_with_their_context(self):
        heading = "1.1 Inspection method"
        paragraphs = self.fixture_paragraphs([heading, "The following items are checked:",
                                               "\u25cf Bearings. \u25cf Seals."], section=heading)
        units = lit.make_units(paragraphs, self.tokenizer)
        groups = lit.baseline_groups(units, self.tokenizer)
        result = lit.context_join_groups(groups, self.tokenizer)
        self.assertEqual(len(result), 1)
        self.assertEqual([j["reason"] for j in result[0]["joined_boundaries"]],
                         ["heading_with_following_text", "list_introduction_with_items"])
        self.assertEqual(result[0]["units"], units)

    def test_context_join_never_exceeds_the_encoder_limit(self):
        paragraphs = self.fixture_paragraphs([" ".join(["measurement"] * 180),
                                               " ".join(["measurement"] * 160) + "."])
        units = lit.make_units(paragraphs, self.tokenizer)
        self.assertEqual(len(units), 2)
        result = lit.context_join_groups([[u] for u in units], self.tokenizer)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["unresolved_boundaries"][0]["blocked_by"], "embedding_token_limit")
        self.assertTrue(all(self.tokenizer.count(lit.joined(r["units"])) <= self.tokenizer.limit for r in result))

    def test_context_join_respects_sections_documents_and_complete_sentences(self):
        paragraphs = self.fixture_paragraphs(["A completed statement.", "another statement follows."])
        units = lit.make_units(paragraphs, self.tokenizer)
        self.assertIsNone(lit.context_join_reason([units[0]], [units[1]]))
        paragraphs = self.fixture_paragraphs(["A partial statement", "continues here."])
        units = lit.make_units(paragraphs, self.tokenizer)
        self.assertIsNotNone(lit.context_join_reason([units[0]], [units[1]]))
        other = deepcopy(units[1])
        other["section"] = "Another section"
        self.assertIsNone(lit.context_join_reason([units[0]], [other]))
        other = deepcopy(units[1])
        other["document_id"] = "another_document"
        self.assertIsNone(lit.context_join_reason([units[0]], [other]))

    def test_synthetic_paper_entrypoint_cache_and_fixture_provenance(self):
        with tempfile.TemporaryDirectory(prefix="phm_literature_test_") as temp:
            temp = Path(temp)
            import pymupdf
            sources = temp / "sources"
            sources.mkdir()
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((60, 100), "1 Inspection method")
                page.insert_text((60, 140), "The motor runs. The current rises. The alarm follows.")
                pdf.save(sources / "synthetic_fixture.pdf")
            policy = temp / "policy.json"
            lit.atomic_json(policy, {"sources": {}})
            shutil.copytree(self.output / "model_assets", temp / "model_assets")
            first = lit.run_stage(sources, temp, policy)
            baseline = (temp / "chunks_baseline.jsonl").read_bytes()
            request = json.loads((temp / "qwen_requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
            lit.atomic_json(temp / "qwen_responses" / (request["request_id"] + ".json"), {
                "origin": "synthetic_fixture", "request_id": request["request_id"],
                "model": lit.QWEN, "revision": lit.QWEN_REVISION,
                "raw_output": json.dumps({"chunk_end_ids": request["unit_ids"]})})
            second = lit.run_stage(sources, temp, policy)
            self.assertEqual(baseline, (temp / "chunks_baseline.jsonl").read_bytes())
            self.assertFalse(first["audits"][0]["extraction_cache_reused"])
            self.assertTrue(second["audits"][0]["extraction_cache_reused"])
            chunks = [json.loads(line) for line in (temp / "chunks_qwen_candidate.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(c["fallback_reason"] == "response_provenance_mismatch" for c in chunks))
            self.assertTrue(all(c["boundary_selection_method"] == "paragraph_fallback" for c in chunks))
            self.assertTrue(all(c["preparation_method"].startswith("paragraph_fallback") for c in chunks))
            self.assertEqual(second["audits"][0]["qwen_validated_requests"], 0)
            self.assertFalse(any(c["diagnostic_approved"] for c in chunks))
            folder = temp / chunks[0]["document_id"]
            paragraphs = [json.loads(line) for line in (folder / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()]
            page1 = [p["text"] for p in paragraphs if p["page"] == 1]
            self.assertEqual(page1, ["1 Inspection method", "The motor runs. The current rises. The alarm follows."])

    def test_excluded_source_is_blocked_before_extraction_or_model_loading(self):
        with tempfile.TemporaryDirectory(prefix="phm_exclusion_test_") as temp:
            temp = Path(temp)
            source = temp / "excluded.pdf"
            source.write_bytes(b"synthetic excluded bytes; deliberately not a PDF")
            key = lit.digest(source.read_bytes())
            policy = temp / "policy.json"
            lit.atomic_json(policy, {"sources": {key: {"filename": source.name, "excluded": True}}})
            with patch.object(lit, "EmbeddingTokenizer") as tokenizer, self.assertRaisesRegex(ValueError, "excluded"):
                lit.run_stage(temp, temp / "out", policy)
            tokenizer.assert_not_called()
            lit.atomic_json(temp / "out/preparation_contract.json", {"sources": [key]})
            with self.assertRaisesRegex(ValueError, "excluded source"):
                lit.check_cached_source_policy(temp / "out", policy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
