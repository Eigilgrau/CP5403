"""Leakage, grounding and resumability checks using temporary synthetic fixtures.

These are software tests, never Qwen evaluation results. Nothing is written into
the actual case bank, diagnostic response cache or research results.
"""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

import phm_diagnosis as diagnosis
from phm_run_reporting import RunReport, atomic_json, atomic_text


class DiagnosisChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="phm_diagnosis_fixture_")
        self.root = Path(self.temp.name)
        self.config = json.loads(Path(__file__).with_name("experiment_config.json").read_text(encoding="utf-8"))
        self.dictionary = {"feature_order": ["ch1_median", "ch2_change"], "notes": ["Synthetic fixture; no physical units."]}
        values = {"ch1_median": 1.23456789, "ch2_change": 4.0}
        self.record = {
            "case_id": "synthetic_query", "lookback_minutes": 180, "model_id": "synthetic_model", "feature_values": values,
            "preparation_flags": {"clipped_features": [], "imputed_features": []},
            "quality": {key: 1.0 for key in diagnosis.QUALITY_KEYS},
            "class_scores": dict(zip(diagnosis.LABELS, [0.1, 0.6, 0.3])), "historical_cases": []}
        for index, label in enumerate(diagnosis.LABELS):
            history_values = {"ch1_median": 1.0, "ch2_change": 8.0}
            self.record["historical_cases"].append({"case_id": f"synthetic_history_{index}", "recorded_outcome": label,
                "distance": index + 0.1, "feature_values": history_values,
                "comparison": [{"comparison": kind, "feature": name, "query_value": values[name],
                    "historical_value": history_values[name], "absolute_prepared_standardised_gap": index + 0.25}
                    for kind, name in [("closest_numeric_feature", "ch1_median"), ("largest_standardised_difference", "ch2_change")]]})
        self.output = {"predicted_class": "RuntimeExcessive", "feature_evidence": [{"feature": "ch1_median", "value": 1.23457}],
            "case_comparison": {"case_id": "synthetic_history_1", "similarity_feature": "ch1_median", "difference_feature": "ch2_change"},
            "literature_citations": [], "explanation": "Synthetic test explanation.", "limitation": "Synthetic test limitation."}

    def tearDown(self):
        self.temp.cleanup()

    def request(self):
        return diagnosis.make_request(self.record, self.dictionary, self.config)

    def test_private_fields_and_self_neighbours_are_rejected(self):
        for key in ("reference_label", "site_id", "decision_time", "split", "episode_id"):
            record = deepcopy(self.record)
            record[key] = "private"
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnosis.make_request(record, self.dictionary, self.config)
        record = deepcopy(self.record)
        record["historical_cases"][0]["case_id"] = record["case_id"]
        with self.assertRaises(ValueError):
            diagnosis.make_request(record, self.dictionary, self.config)
        text = json.dumps(self.request()["messages"])
        self.assertNotIn("reference_label", text)
        self.assertNotIn("site_id", text)
        self.assertNotIn("decision_time", text)

    def test_compact_prompt_preserves_all_features_and_history_order(self):
        request = self.request()
        payload = json.loads(request["messages"][1]["content"].split("\n", 1)[1])
        self.assertEqual(payload["feature_order"], self.dictionary["feature_order"])
        self.assertEqual(payload["query"]["feature_values"], [1.23457, 4.0])
        self.assertEqual([c["case_id"] for c in payload["historical_cases"]], [c["case_id"] for c in self.record["historical_cases"]])
        self.assertTrue(all(len(c["feature_values"]) == 2 for c in payload["historical_cases"]))
        self.assertEqual(request["evidence_sha256"], diagnosis.fingerprint(self.record))

    def test_D_cannot_silently_run_without_literature(self):
        with self.assertRaisesRegex(ValueError, "D requires"):
            diagnosis.make_request(self.record, self.dictionary, self.config, condition="D")

    def test_portable_bundle_hashes_survive_windows_and_labels_are_not_opened(self):
        texts = {"validation_evidence.jsonl": json.dumps(self.record) + "\n",
                 "feature_dictionary.json": json.dumps(self.dictionary) + "\n"}
        for name, text in texts.items():
            atomic_text(self.root / name, text)
            self.assertEqual((self.root / name).read_bytes(), text.encode("utf-8"))
        atomic_json(self.root / "export_contract.json", {"scope": "validation_pilot", "historical_k": 3,
            "case_ids": [self.record["case_id"]], "files": {name: diagnosis.fingerprint(text.encode()) for name, text in texts.items()}})
        # No scoring label file exists: prompt preparation must still work.
        _, records, requests = diagnosis.prepare_requests(self.root, self.config, 1)
        self.assertEqual(records, [self.record])
        self.assertEqual(len(requests), 1)
        atomic_text(self.root / "validation_evidence.jsonl", texts["validation_evidence.jsonl"] + "\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            diagnosis.load_bundle(self.root)

    def test_numeric_citations_and_comparisons_are_checked(self):
        self.assertTrue(diagnosis.validate_response(json.dumps(self.output), self.record)["valid"])
        variants = []
        wrong_number = deepcopy(self.output)
        wrong_number["feature_evidence"][0]["value"] = 1.23
        variants.append(wrong_number)
        invented = deepcopy(self.output)
        invented["case_comparison"]["case_id"] = "invented"
        variants.append(invented)
        swapped = deepcopy(self.output)
        swapped["case_comparison"]["difference_feature"] = "ch1_median"
        variants.append(swapped)
        literature = deepcopy(self.output)
        literature["literature_citations"] = ["invented_paper"]
        variants.append(literature)
        nonnumeric = deepcopy(self.output)
        nonnumeric["feature_evidence"][0]["value"] = True
        variants.append(nonnumeric)
        for output in variants:
            with self.subTest(output=output):
                self.assertFalse(diagnosis.validate_response(json.dumps(output), self.record)["valid"])

    def test_invalid_json_and_duplicate_keys_do_not_get_repaired(self):
        for text in ("not JSON", "```json\n{}\n```", "[]", '{"a":1,"a":2}', '{"a":NaN}'):
            with self.subTest(text=text):
                self.assertFalse(diagnosis.validate_response(text, self.record)["valid"])
        abstain = deepcopy(self.output)
        abstain.update(predicted_class="InsufficientEvidence", feature_evidence=[], case_comparison=None)
        self.assertTrue(diagnosis.validate_response(json.dumps(abstain), self.record)["valid"])

    def test_cache_identity_changes_with_evidence_prompt_or_generation_settings(self):
        first = self.request()["request_id"]
        self.assertEqual(first, self.request()["request_id"])
        config = deepcopy(self.config)
        config["generator"]["max_new_tokens"] += 1
        self.assertNotEqual(first, diagnosis.make_request(self.record, self.dictionary, config)["request_id"])
        dictionary = deepcopy(self.dictionary)
        dictionary["notes"].append("Changed notes.")
        self.assertNotEqual(first, diagnosis.make_request(self.record, dictionary, self.config)["request_id"])
        with patch.object(diagnosis, "SYSTEM_PROMPT", "Changed prompt"):
            self.assertNotEqual(first, self.request()["request_id"])

    def envelope(self):
        request = self.request()
        raw = json.dumps(self.output)
        # Deliberately simulates an actual envelope inside this temporary test directory only.
        return {"origin": "model_inference", "raw_output": raw, "raw_output_sha256": diagnosis.fingerprint(raw), "error": None,
                **{key: request[key] for key in ("request_id", "case_id", "condition", "generator", "evidence_sha256")}}

    def test_cache_tampering_and_synthetic_provenance_are_rejected(self):
        path = self.root / "cache.json"
        envelope = self.envelope()
        atomic_json(path, envelope)
        self.assertEqual(diagnosis.inspect_cache(path, self.request(), self.record)[0]["status"], "valid")
        for key, value in (("raw_output", "{}"), ("origin", "synthetic_fixture"), ("case_id", "wrong"), ("fixture_only", True)):
            changed = {**envelope, key: value}
            atomic_json(path, changed)
            self.assertEqual(diagnosis.inspect_cache(path, self.request(), self.record)[0]["status"], "invalid_cache")

    def test_resume_does_not_load_the_model_for_completed_requests(self):
        import phm_diagnosis_qwen as runner
        request = self.request()
        cache = self.root / "responses"
        atomic_json(cache / (request["request_id"] + ".json"), self.envelope())
        report = RunReport(self.root, "synthetic_resume_check")
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            result = runner._infer([request], [self.record], cache, self.config, report, False)
        self.assertEqual(result["new_responses"], 0)
        report.finish()

    def test_model_setup_failure_is_saved(self):
        import phm_diagnosis_qwen as runner
        (self.root / "Files").mkdir()
        atomic_json(self.root / "Files/experiment_config.json", self.config)
        report = RunReport(self.root, "synthetic_setup_failure")
        with patch.object(runner, "prepare_requests", return_value=({"scope": "validation_pilot"}, [self.record], [self.request()])), \
             patch.object(runner, "_infer", side_effect=RuntimeError("synthetic GPU failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic GPU failure"):
                runner.run_condition_C(self.root, reporter=report)
        self.assertEqual(report.data["status"], "failed")
        self.assertIn("synthetic GPU failure", (report.folder / "traceback.txt").read_text())
        self.assertFalse(report.data["diagnostic_experiment_executed"])
        report.finish()

    def test_pilot_scores_use_separate_labels_and_report_incomplete_outputs(self):
        request = self.request()
        cache = self.root / "responses"
        envelope = self.envelope()
        envelope["raw_output"] = "bad JSON"
        envelope["raw_output_sha256"] = diagnosis.fingerprint(envelope["raw_output"])
        atomic_json(cache / (request["request_id"] + ".json"), envelope)
        bundle = self.root / "bundle"
        bundle.mkdir()
        labels_path = bundle / "private_validation_labels.jsonl"
        labels_path.write_text(json.dumps({"case_id": self.record["case_id"], "reference_label": "RuntimeExcessive"}) + "\n", encoding="utf-8")
        atomic_json(bundle / "export_contract.json", {"files": {labels_path.name: diagnosis.fingerprint(labels_path.read_bytes())}})
        report = RunReport(self.root, "synthetic_scoring_check")
        metrics = diagnosis.capture_diagnostics(report, bundle, cache, [request], [self.record], score=True)
        self.assertEqual(metrics["accuracy_over_attempted"], 0)
        self.assertEqual(metrics["status_counts"], {"invalid_response": 1})
        report.finish()
        self.assertEqual(report.data["status"], "completed_with_issues")
        self.assertTrue(report.data["diagnostic_inference_observed"])
        self.assertFalse(report.data["diagnostic_experiment_executed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
