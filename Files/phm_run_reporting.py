"""Persist notebook progress and reviewable results to the synced project directory.

Only the standard library is required for logging. DuckDB is imported lazily when
snapshotting existing numerical artifacts. No model weights or environment
variables are copied into reports. A completed preparation run is not a completed
C/D diagnostic experiment.
"""

from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import traceback


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def redact(value):
    text = str(value)
    text = re.sub(r"hf_[A-Za-z0-9]{8,}", "[REDACTED_HF_TOKEN]", text)
    return re.sub(r"(?i)(\b(?:access_token|api_key|password|authorization)\b\s*[:=]\s*)([^\s,;]+)",
                  r"\1[REDACTED]", text)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(sanitise(value), indent=2, ensure_ascii=False, allow_nan=False,
                                default=str) + "\n")


def sanitise(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise(v) for v in value]
    return value


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_experiment_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    conditions = config.get("conditions", {})
    if set(conditions) != {"C", "D"}:
        raise ValueError("The active experiment must contain C and D only.")
    for name, literature in (("C", False), ("D", True)):
        if conditions[name].get("historical_cases") is not True or conditions[name].get("literature") is not literature:
            raise ValueError("Both conditions require cases; only D may include literature.")
    required = {"same_query_ids", "same_features_and_regression_scores",
                "same_historical_case_ids_and_order", "same_generator_and_decoding",
                "query_labels_outside_prompts_and_retrieval"}
    if any(config.get("paired_controls", {}).get(key) is not True for key in required):
        raise ValueError("C/D pairing and label-separation controls must remain enabled.")
    if config.get("evaluation", {}).get("diagnostic_generations_per_query") != 2:
        raise ValueError("C/D requires two diagnostic generations per query.")
    return config


class _Tee:
    def __init__(self, target, report):
        self.target, self.report = target, report

    def write(self, text):
        self.target.write(text)
        self.report.log(text)
        return len(text)

    def flush(self):
        self.target.flush()

    def __getattr__(self, name):
        return getattr(self.target, name)


class RunReport:
    def __init__(self, project_root, stage):
        self.root = Path(project_root).resolve()
        safe_stage = re.sub(r"[^A-Za-z0-9_-]+", "_", stage)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + safe_stage + "_" + uuid4().hex[:8]
        self.folder = self.root / "Outputs" / "runs" / self.run_id
        self.folder.mkdir(parents=True, exist_ok=False)
        self._hooks = None
        self._capture = None
        self.data = {"schema_version": "0.6.0", "run_id": self.run_id, "stage": stage,
                     "status": "running", "started_utc": utc_now(), "updated_utc": utc_now(),
                     "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                     "execution": "colab" if "google.colab" in sys.modules else "local"},
                     "steps": [], "checks": [], "notes": [], "artifacts": [], "metrics": {},
                     "diagnostic_experiment_executed": False, "diagnostic_inference_observed": False,
                     "remaining_pipeline": ["condition C validation pilot and review", "approved literature retrieval for D",
                                            "paired C/D validation", "held-out paired C/D evaluation"]}
        self.refresh_packages()
        self.save()

    def refresh_packages(self):
        versions = {}
        for name in ("duckdb", "numpy", "pandas", "scikit-learn", "joblib", "PyMuPDF", "tokenizers",
                     "huggingface-hub", "torch", "transformers", "accelerate", "bitsandbytes"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.data["environment"]["packages"] = versions

    def log(self, text):
        with (self.folder / "console.log").open("a", encoding="utf-8") as handle:
            handle.write(redact(text))

    def event(self, kind, **fields):
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitise({"utc": utc_now(), "event": kind, **fields}), default=str) + "\n")

    def note(self, message):
        if message not in self.data["notes"]:
            self.data["notes"].append(message)
        self.save()

    def check(self, name, passed, detail=""):
        self.data["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        self.save()
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    def configure(self, path):
        with self.step("load_C_D_configuration"):
            config = load_experiment_config(path)
            self.data["experiment"] = config
            atomic_json(self.folder / "experiment_config.json", config)
            self.add_artifact(path)
        return config

    def add_artifact(self, path):
        path = Path(path).resolve()
        try:
            label = path.relative_to(self.root).as_posix()
        except ValueError:
            label = str(path)
        record = {"path": label, "exists": path.is_file()}
        if record["exists"]:
            record.update(bytes=path.stat().st_size, modified_utc=datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc).isoformat())
            if path.stat().st_size <= 10 * 1024 * 1024:
                record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.data["artifacts"] = [r for r in self.data["artifacts"] if r["path"] != label] + [record]

    @contextmanager
    def capture_output(self):
        if isinstance(sys.stdout, _Tee) and sys.stdout.report is self:
            yield
        else:
            with redirect_stdout(_Tee(sys.stdout, self)), redirect_stderr(_Tee(sys.stderr, self)):
                yield

    @contextmanager
    def step(self, name):
        if self.data["status"] != "running":
            raise RuntimeError("Start a new report by rerunning the notebook setup cell before continuing.")
        entry = {"name": name, "status": "running", "started_utc": utc_now()}
        self.data["steps"].append(entry)
        self.event("step_started", name=name)
        self.save()
        started = perf_counter()
        try:
            with self.capture_output():
                yield
        except BaseException as exc:
            entry.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         seconds=perf_counter() - started)
            self.fail(exc)
            raise
        else:
            entry.update(status="completed", seconds=perf_counter() - started)
            self.event("step_completed", name=name, seconds=entry["seconds"])
            self.save()

    def fail(self, exc):
        if self.data["status"] in {"failed", "interrupted"}:
            return
        self.data.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=redact(f"{type(exc).__name__}: {exc}"), finished_utc=utc_now())
        atomic_text(self.folder / "traceback.txt", redact("".join(traceback.format_exception(exc))))
        self.event("run_failed", error=self.data["error"])
        self.save()

    def run_command(self, args, name):
        with self.step(name):
            self.event("command", arguments=[redact(x) for x in args])
            with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="")
                returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, args)

    def install_notebook_hooks(self, shell):
        """Capture errors even in cells that do not use step(), including imports."""
        if shell is None:
            return
        previous = getattr(shell, "_phm_run_report", None)
        if previous is not None:
            if previous.data["status"] == "running":
                previous.data["status"] = "interrupted"
                previous.data["finished_utc"] = utc_now()
                previous.note("Notebook setup was rerun before this run was finished.")
            previous.detach_notebook_hooks()

        def before_cell(info):
            if self.data["status"] != "running":
                return
            self._capture = ExitStack()
            self._capture.enter_context(self.capture_output())
            self.event("cell_started", cell_id=getattr(info, "cell_id", None))

        def after_cell(result):
            if self._capture is not None:
                self._capture.close()
                self._capture = None
            error = getattr(result, "error_before_exec", None) or getattr(result, "error_in_exec", None)
            if error is not None:
                self.fail(error)
            elif self.data["status"] == "running":
                self.event("cell_completed", execution_count=getattr(result, "execution_count", None))
                self.save()

        shell.events.register("pre_run_cell", before_cell)
        shell.events.register("post_run_cell", after_cell)
        shell._phm_run_report = self
        self._hooks = (shell, before_cell, after_cell)
        self.save()

    def detach_notebook_hooks(self):
        if self._capture is not None:
            self._capture.close()
            self._capture = None
        if self._hooks is not None:
            shell, before, after = self._hooks
            shell.events.unregister("pre_run_cell", before)
            shell.events.unregister("post_run_cell", after)
            if getattr(shell, "_phm_run_report", None) is self:
                shell._phm_run_report = None
            self._hooks = None

    def capture_literature(self, output_dir):
        """Snapshot current jobs; never treat raw or synthetic responses as validated chunks."""
        folder = Path(output_dir)
        contract_path = folder / "preparation_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
        requests = read_rows(folder / "qwen_requests.jsonl") if (folder / "qwen_requests.jsonl").exists() else []
        chunks = read_rows(folder / "chunks_qwen_candidate.jsonl") if (folder / "chunks_qwen_candidate.jsonl").exists() else []
        audit = {}
        for item in contract.get("audits", []):
            path = folder / item["document_id"] / "request_audit.jsonl"
            if path.exists():
                audit.update({r["request_id"]: r for r in read_rows(path)})
        rows, response_sample = [], []
        for request in requests:
            rid = request["request_id"]
            cache = folder / "qwen_responses" / (rid + ".json")
            row = {"request_id": rid, "status": "not_run", "reason": "", "input_tokens": None,
                   "output_tokens": None, "seconds": None, "generation_hit_limit": None}
            if cache.exists():
                try:
                    response = json.loads(cache.read_text(encoding="utf-8"))
                    if (not isinstance(response, dict) or response.get("origin") != "model_inference"
                        or any(response.get(k) != request.get(k) for k in ("request_id", "model", "revision"))):
                        raise ValueError("response_provenance_mismatch")
                    row.update({k: response.get(k) for k in ("input_tokens", "output_tokens", "seconds", "generation_hit_limit")})
                    if response.get("error"):
                        row.update(status="runtime_error", reason=redact(response["error"]))
                    elif not contract_path.exists() or cache.stat().st_mtime > contract_path.stat().st_mtime or rid not in audit:
                        row["status"] = "awaiting_validation"
                    else:
                        row.update(status=audit[rid]["status"], reason=audit[rid].get("reason") or "")
                    if len(response_sample) < 3:
                        response_sample.append({"request_id": rid, "status": row["status"],
                                                "raw_output": response.get("raw_output"), "error": response.get("error")})
                except (ValueError, TypeError) as exc:
                    row.update(status="invalid_cache", reason=str(exc))
            rows.append(row)
        counts = dict(Counter(r["status"] for r in rows))
        status = {"preparation_contract_utc": contract.get("created_utc"),
                  "documents": len(contract.get("sources", [])), "requests_total": len(requests),
                  "request_status_counts": counts, "candidate_chunks": len(chunks),
                  "candidate_methods": dict(Counter(c["preparation_method"] for c in chunks)),
                  "boundary_postprocessing_version": contract.get("boundary_postprocessing_version"),
                  "python_context_joined_chunks": sum(bool(c.get("boundary_cleanup", {}).get("joined_boundaries")) for c in chunks),
                  "unresolved_context_boundaries": sum(len(c.get("boundary_cleanup", {}).get("unresolved_boundaries", [])) for c in chunks),
                  "diagnostic_approved_chunks": sum(bool(c.get("diagnostic_approved")) for c in chunks),
                  "diagnostic_index_built": bool(contract.get("diagnostic_index_built", False)),
                  "qwen_validated_requests_in_last_preparation": sum(a.get("qwen_validated_requests", 0) for a in contract.get("audits", []))}
        self.data["metrics"]["literature"] = status
        self.write_csv("qwen_request_status.csv", rows)
        atomic_json(self.folder / "literature_sample.json", {"scope": "inspection only; not an evaluation score",
                    "chunks": chunks[:3], "actual_response_samples": response_sample})
        atomic_json(self.folder / "literature_status.json", status)
        if contract:
            atomic_json(self.folder / "preparation_contract_snapshot.json", contract)
        for name in ("inventory.jsonl", "chunks_baseline.jsonl", "chunks_qwen_candidate.jsonl", "qwen_requests.jsonl",
                     "preparation_contract.json", "qwen_runtime.json", "chunk_review_template.csv"):
            self.add_artifact(folder / name)
        runtime = folder / "qwen_runtime.json"
        if runtime.exists():
            atomic_json(self.folder / "qwen_runtime_snapshot.json", json.loads(runtime.read_text(encoding="utf-8")))
        self.save()
        return status

    def write_csv(self, name, rows):
        if not rows:
            atomic_text(self.folder / name, "")
            return
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(self.folder / name, redact(stream.getvalue()))

    def capture_numerical(self):
        """Export small existing contracts/aggregate metrics, never unpickle or refit a model."""
        import duckdb
        selection = {"case_inputs.duckdb": [],
                     "case_design.duckdb": ["case_contract", "assumption_audit", "fold_metrics", "coverage_sensitivity"],
                     "case_system.duckdb": ["case_system_contract", "validation_metrics", "feature_schema"]}
        snapshots = []
        for filename, tables in selection.items():
            path = self.root / "Datasets" / "PHM15-prepared" / filename
            self.add_artifact(path)
            if not path.exists():
                continue
            item = {"file": filename, "scope": "existing artifact snapshot; no model execution", "tables": []}
            with duckdb.connect(str(path), read_only=True) as db:
                available = {r[0] for r in db.execute("show tables").fetchall()}
                for table in tables:
                    if table not in available:
                        continue
                    cursor = db.execute(f'SELECT * FROM "{table}" LIMIT 2001')
                    columns = [c[0] for c in cursor.description]
                    values = cursor.fetchall()
                    self.write_csv(f"{path.stem}_{table}.csv", [dict(zip(columns, row)) for row in values[:2000]])
                    item["tables"].append({"name": table, "exported_rows": min(2000, len(values)), "truncated": len(values) > 2000})
            snapshots.append(item)
        self.data["metrics"]["numerical_snapshots"] = snapshots
        self.save()

    def finish(self):
        if self.data["status"] == "running":
            counts = self.data["metrics"].get("literature", {}).get("request_status_counts", {})
            has_issues = any(counts.get(k, 0) for k in ("runtime_error", "invalid_cache", "paragraph_fallback"))
            diagnostic_counts = self.data["metrics"].get("diagnostics", {}).get("status_counts", {})
            has_issues |= any(diagnostic_counts.get(k, 0) for k in ("runtime_error", "invalid_cache", "invalid_response", "not_run"))
            self.data.update(status="completed_with_issues" if has_issues else "completed", finished_utc=utc_now())
        self.refresh_packages()
        self.event("run_finished", status=self.data["status"])
        self.save()
        self.detach_notebook_hooks()
        return self.folder

    def save(self):
        self.data["updated_utc"] = utc_now()
        atomic_json(self.folder / "run_report.json", self.data)
        lines = ["# PHM run review", "", f"Run: `{self.run_id}`", f"Stage: **{self.data['stage']}**",
                 f"Status: **{self.data['status']}**", f"Environment: **{self.data['environment']['execution']}**",
                 f"Updated (UTC): {self.data['updated_utc']}", "",
                 "Active diagnostic experiment: **C (cases) vs D (same cases + literature)**.",
                 ("Condition C validation responses are available. This is not the completed paired C/D test experiment."
                  if self.data.get("diagnostic_inference_observed") else
                  "This report covers the named stage. No diagnostic inference is recorded for this run."), "",
                 "A `running` status with no new events may indicate a stopped runtime. Check Colab; a hard disconnect cannot write a final error.", "",
                 "## Steps", "", "| Step | Status | Seconds |", "|---|---|---|"]
        for step in self.data["steps"]:
            seconds = f"{step['seconds']:.2f}" if "seconds" in step else ""
            lines.append(f"| {step['name']} | {step['status']} | {seconds} |")
        if "error" in self.data:
            lines += ["", "## Error", "", self.data["error"], "", "See `traceback.txt` and `console.log` in this run folder."]
        if self.data["checks"]:
            lines += ["", "## Checks", ""] + [f"- {'PASS' if c['passed'] else 'FAIL'}: {c['name']} {c['detail']}" for c in self.data["checks"]]
        lit = self.data["metrics"].get("literature")
        if lit:
            lines += ["", "## Literature status", "", f"- Documents: {lit['documents']}; candidate chunks: {lit['candidate_chunks']}.",
                      f"- Qwen request statuses: {json.dumps(lit['request_status_counts'])}.",
                      f"- Candidate chunk methods: {json.dumps(lit['candidate_methods'])}.",
                      f"- Python context cleanup: {lit.get('python_context_joined_chunks', 0)} joined chunks; {lit.get('unresolved_context_boundaries', 0)} detected boundaries left for review.",
                      "- Methods ending in `_context_join` include audited Python joins; raw Qwen acceptance/fallback counts remain separate.",
                      f"- Approved diagnostic chunks: {lit['diagnostic_approved_chunks']}.",
                      f"- Diagnostic literature index built: {lit['diagnostic_index_built']}.",
                      "- Raw cached responses still awaiting validation are not counted as successful chunks."]
        diagnosis = self.data["metrics"].get("diagnostics")
        if diagnosis:
            lines += ["", "## Condition C validation pilot", "",
                      f"- Selected queries: {diagnosis['selected_queries']}; statuses: {json.dumps(diagnosis['status_counts'])}.",
                      "- D has not run. Final held-out paired evaluation remains outstanding.",
                      "- Automatic JSON, citation and numeric checks do not assess full explanation support or replace human review.",
                      "- Inspect `diagnostic_requests.jsonl`, `diagnostic_responses.json` and `diagnostic_status.csv`."]
            if "accuracy_over_attempted" in diagnosis:
                lines += [f"- Development accuracy over attempted queries: {diagnosis['accuracy_over_attempted']}.",
                          "- This small, balanced validation pilot is a workflow check, not a test-set performance claim."]
        if self.data["notes"]:
            lines += ["", "## Notes", ""] + ["- " + n for n in self.data["notes"]]
        lines += ["", "## Files to inspect", "", "- `run_report.json`: runtime, experiment configuration, checks, metrics and file fingerprints.",
                  "- `console.log`, `events.jsonl`, and `traceback.txt` when present: progress and failures.",
                  "- `qwen_request_status.csv` and `literature_sample.json` when present: actual chunking outcomes.",
                  "- Numerical CSVs when present: existing saved contracts and development metrics.", "",
                  "## Remaining pipeline work", ""] + ["- " + x for x in self.data["remaining_pipeline"]]
        summary = redact("\n".join(lines)) + "\n"
        atomic_text(self.folder / "SUMMARY.md", summary)
        relative = self.folder.relative_to(self.root).as_posix()
        atomic_json(self.root / "Outputs" / "LATEST_RUN.json", {"run_id": self.run_id, "run_directory": relative,
                    "status": self.data["status"], "updated_utc": self.data["updated_utc"]})
        atomic_text(self.root / "Outputs" / "LATEST_RUN.md", summary + f"\nRun directory relative to project root: `{relative}`\n")
