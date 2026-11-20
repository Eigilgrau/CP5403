"""Exercise failed runs, notebook hooks, C/D controls and cache provenance.

All response fixtures live in temporary directories. They are never saved into
the real corpus or reported as actual Qwen results.
"""

from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import os
import tempfile
import unittest

from phm_run_reporting import RunReport, atomic_json, load_experiment_config, read_rows


class ReportingChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="phm_reporting_fixture_")
        self.root = Path(self.temp.name)
        self.config = json.loads(Path(__file__).with_name("experiment_config.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def test_only_C_D_with_matching_case_evidence_is_allowed(self):
        path = self.root / "config.json"
        atomic_json(path, self.config)
        self.assertEqual(set(load_experiment_config(path)["conditions"]), {"C", "D"})
        invalid = []
        extra = deepcopy(self.config)
        extra["conditions"]["B"] = {"historical_cases": False, "literature": False}
        invalid.append(extra)
        unequal = deepcopy(self.config)
        unequal["conditions"]["D"]["historical_cases"] = False
        invalid.append(unequal)
        control = deepcopy(self.config)
        control["paired_controls"]["same_historical_case_ids_and_order"] = False
        invalid.append(control)
        for config in invalid:
            atomic_json(path, config)
            with self.assertRaises(ValueError):
                load_experiment_config(path)

    def test_failure_persists_traceback_and_redacts_tokens_without_final_cell(self):
        report = RunReport(self.root, "synthetic_failure_check")
        token = "hf_SyntheticSecret12345678"
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            with report.step("dependency_or_model_loading"):
                print("token=" + token)
                raise RuntimeError("Fixture download failure: " + token)
        data = json.loads((report.folder / "run_report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "failed")
        self.assertFalse(data["diagnostic_experiment_executed"])
        self.assertIn("RuntimeError", (report.folder / "traceback.txt").read_text(encoding="utf-8"))
        for p in report.folder.glob("*"):
            self.assertNotIn(token, p.read_text(encoding="utf-8"))
        self.assertEqual(json.loads((self.root / "Outputs/LATEST_RUN.json").read_text())["status"], "failed")
        self.assertTrue(read_rows(report.folder / "events.jsonl"))
        report.finish()
        self.assertEqual(report.data["status"], "failed")

    def test_model_setup_error_is_captured_by_runner(self):
        import phm_literature_qwen as runner
        report = RunReport(self.root, "synthetic_model_setup_check")
        with patch.object(runner, "_run_qwen", side_effect=ImportError("fixture missing GPU dependency")):
            with self.assertRaises(ImportError):
                runner.run_qwen(self.root / "empty_literature", max_requests=3, reporter=report)
        self.assertEqual(report.data["status"], "failed")
        self.assertIn("missing GPU dependency", (report.folder / "traceback.txt").read_text())

    def test_notebook_error_hook_and_new_run_restore_output_stream(self):
        class Events:
            def __init__(self):
                self.handlers = {"pre_run_cell": [], "post_run_cell": []}
            def register(self, key, callback):
                self.handlers[key].append(callback)
            def unregister(self, key, callback):
                self.handlers[key].remove(callback)
        shell = SimpleNamespace(events=Events())
        report = RunReport(self.root, "synthetic_notebook_check")
        report.install_notebook_hooks(shell)
        with redirect_stdout(io.StringIO()) as sink:
            shell.events.handlers["pre_run_cell"][0](SimpleNamespace(cell_id="fixture"))
            print("ordinary cell output")
            shell.events.handlers["post_run_cell"][0](SimpleNamespace(error_before_exec=None,
                error_in_exec=ValueError("fixture cell failure"), execution_count=2))
            print("stream restored")
            self.assertIn("stream restored", sink.getvalue())
        self.assertEqual(report.data["status"], "failed")
        self.assertIn("ordinary cell output", (report.folder / "console.log").read_text())
        next_report = RunReport(self.root, "synthetic_second_run")
        next_report.install_notebook_hooks(shell)
        self.assertEqual(len(shell.events.handlers["post_run_cell"]), 1)
        next_report.finish()
        self.assertEqual(shell.events.handlers["post_run_cell"], [])
        self.assertNotEqual(report.folder, next_report.folder)

    def make_literature_fixture(self):
        folder = self.root / "fixture_literature"
        folder.mkdir()
        requests = [{"request_id": f"fixture_{i}", "model": "fixture_model", "revision": "fixture_revision"} for i in range(3)]
        (folder / "qwen_requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests), encoding="utf-8")
        atomic_json(folder / "preparation_contract.json", {"sources": ["fixture"], "audits": [], "diagnostic_index_built": False})
        (folder / "chunks_qwen_candidate.jsonl").write_text(json.dumps({"preparation_method": "paragraph_fallback",
            "text": "Synthetic fixture text.", "diagnostic_approved": False}) + "\n", encoding="utf-8")
        return folder, requests

    def test_missing_and_synthetic_responses_never_become_Qwen_successes(self):
        folder, requests = self.make_literature_fixture()
        atomic_json(folder / "qwen_responses/fixture_0.json", {**requests[0], "origin": "synthetic_fixture", "raw_output": "{}"})
        report = RunReport(self.root, "synthetic_provenance_check")
        status = report.capture_literature(folder)
        self.assertEqual(status["request_status_counts"], {"invalid_cache": 1, "not_run": 2})
        self.assertEqual(status["diagnostic_approved_chunks"], 0)
        self.assertEqual(status["qwen_validated_requests_in_last_preparation"], 0)
        self.assertEqual(json.loads((report.folder / "literature_sample.json").read_text())["actual_response_samples"], [])
        report.finish()
        self.assertEqual(report.data["status"], "completed_with_issues")

    def test_new_raw_cache_requires_validation_and_runtime_error_is_visible(self):
        folder, requests = self.make_literature_fixture()
        # Simulated cache envelopes exist only in this temporary fixture directory.
        atomic_json(folder / "qwen_responses/fixture_0.json", {**requests[0], "origin": "model_inference",
                    "raw_output": "{}", "error": None, "fixture_only": True})
        atomic_json(folder / "qwen_responses/fixture_1.json", {**requests[1], "origin": "model_inference",
                    "raw_output": None, "error": "fixture out of memory", "fixture_only": True})
        os.utime(folder / "preparation_contract.json", (1000, 1000))
        report = RunReport(self.root, "synthetic_pending_validation_check")
        status = report.capture_literature(folder)
        self.assertEqual(status["request_status_counts"], {"awaiting_validation": 1, "runtime_error": 1, "not_run": 1})
        report.finish()
        self.assertEqual(report.data["status"], "completed_with_issues")

    def test_interruption_retains_a_distinct_status(self):
        report = RunReport(self.root, "synthetic_interruption_check")
        with self.assertRaises(KeyboardInterrupt):
            with report.step("interrupted_step"):
                raise KeyboardInterrupt()
        report.finish()
        self.assertEqual(report.data["status"], "interrupted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
