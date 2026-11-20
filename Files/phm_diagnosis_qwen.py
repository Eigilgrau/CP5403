"""Run the case-only Qwen validation pilot on Colab, saving every response to Drive."""

from pathlib import Path
from time import perf_counter
from uuid import uuid4
import argparse
import importlib.metadata
import json
import platform
import shutil

from phm_diagnosis import capture_diagnostics, fingerprint, inspect_cache, prepare_requests, validate_response
from phm_run_reporting import RunReport, atomic_json, atomic_text, load_experiment_config, utc_now


def _infer(requests, records, cache_dir, config, reporter, retry_errors):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for request, record in zip(requests, records, strict=True):
        checked, _ = inspect_cache(cache_dir / (request["request_id"] + ".json"), request, record)
        if checked["status"] == "not_run" or (retry_errors and checked["status"] in {"runtime_error", "invalid_response", "invalid_cache"}):
            pending.append((request, record))
    if not pending:
        print("Selected requests are cached. Reviewing saved responses; no model load needed.")
        return {"new_responses": 0}

    generator = config["generator"]
    if generator["do_sample"] is not False or generator["quantisation"] != "NF4 4-bit, double quantisation, float16 compute":
        raise ValueError("This runner implements greedy NF4 inference only; review settings before changing the contract.")
    for name in ("max_input_tokens", "max_new_tokens"):
        if type(generator[name]) is not int or generator[name] < 1:
            raise ValueError("Token budgets must be positive integers.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. In Colab select Runtime > Change runtime type > T4 GPU, then rerun setup.")
    set_seed(generator["seed"])
    start = perf_counter()
    reporter.event("tokenizer_loading", model=generator["model"], revision=generator["revision"])
    tokenizer = AutoTokenizer.from_pretrained(generator["model"], revision=generator["revision"])
    token_counts = []
    for request, _ in pending:
        count = len(tokenizer.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True))
        token_counts.append({"case_id": request["case_id"], "request_id": request["request_id"], "input_tokens": count})
    reporter.write_csv("prompt_token_counts.csv", token_counts)
    reporter.check("all_prompts_fit_without_truncation", all(r["input_tokens"] <= generator["max_input_tokens"] for r in token_counts),
                   f"Largest: {max(r['input_tokens'] for r in token_counts)}; limit: {generator['max_input_tokens']}")
    quantisation = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.float16)
    reporter.event("model_loading", model=generator["model"], revision=generator["revision"])
    model = AutoModelForCausalLM.from_pretrained(generator["model"], revision=generator["revision"],
        quantization_config=quantisation, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    runtime = {"model": generator["model"], "revision": generator["revision"], "python": platform.python_version(),
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
               "packages": {name: importlib.metadata.version(name) for name in
                            ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers", "huggingface-hub")},
               "quantisation": generator["quantisation"], "model_load_seconds": perf_counter() - start,
               "device_map": {key: str(value) for key, value in getattr(model, "hf_device_map", {}).items()},
               "generator": generator}
    reporter.data["environment"]["qwen_runtime"] = runtime
    atomic_json(reporter.folder / "qwen_runtime.json", runtime)
    reporter.refresh_packages()
    reporter.save()
    completed = 0
    try:
        for request, record in pending:
            path = cache_dir / (request["request_id"] + ".json")
            if path.exists():
                archived = cache_dir / "previous_attempts" / (request["request_id"] + "_" + uuid4().hex[:8] + ".json")
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, archived)
                reporter.event("previous_attempt_archived", path=str(archived))
            response = {"origin": "model_inference", "started_utc": utc_now(), "runtime": runtime,
                        **{key: request[key] for key in ("request_id", "case_id", "condition", "generator", "evidence_sha256")},
                        "raw_output": None, "error": None}
            reporter.event("diagnostic_request_started", request_id=request["request_id"], case_id=request["case_id"], condition="C")
            reporter.data["metrics"]["diagnostic_progress"] = {"status": "generating", "case_id": request["case_id"], "completed_this_run": completed}
            reporter.save()
            request_started = perf_counter()
            inputs = generated = tail = None
            try:
                inputs = tokenizer.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt")
                response["input_tokens"] = int(inputs["input_ids"].shape[-1])
                if response["input_tokens"] > generator["max_input_tokens"]:
                    raise ValueError("Prompt exceeds budget; no evidence truncation is permitted.")
                inputs = inputs.to(model.device)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                generation_started = perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=generator["max_new_tokens"],
                                               pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                response["generation_seconds"] = perf_counter() - generation_started
                tail = generated[0, response["input_tokens"]:]
                response["output_tokens"] = int(len(tail))
                response["raw_output"] = tokenizer.decode(tail, skip_special_tokens=True)
                response["generation_hit_limit"] = len(tail) >= generator["max_new_tokens"]
                response["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
                response["validation_at_generation"] = validate_response(response["raw_output"], record)
            except (RuntimeError, ValueError) as exc:
                response["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                del inputs, generated, tail
            response.update(request_seconds=perf_counter() - request_started, finished_utc=utc_now(),
                            raw_output_sha256=fingerprint(response["raw_output"]))
            atomic_json(path, response)
            completed += 1
            reporter.event("diagnostic_response_cached", request_id=request["request_id"], error=response["error"],
                           generation_seconds=response.get("generation_seconds"), input_tokens=response.get("input_tokens"),
                           output_tokens=response.get("output_tokens"))
            capture_diagnostics(reporter, reporter.root / "Datasets/PHM15-diagnosis/validation_pilot_v1", cache_dir, requests, records)
            checked = response.get("validation_at_generation", {})
            print(record["case_id"][:12], response["error"] or ("valid JSON and cited fields" if checked.get("valid") else "saved; needs review"), flush=True)
            if response["error"]:
                torch.cuda.empty_cache()
                if "out of memory" in response["error"].lower():
                    break
    finally:
        del model
        torch.cuda.empty_cache()
    reporter.data["metrics"]["diagnostic_progress"] = {"status": "finished", "completed_this_run": completed}
    reporter.save()
    return {"new_responses": completed, "runtime": runtime}


def run_condition_C(project_root, max_cases=3, retry_errors=False, reporter=None):
    root = Path(project_root).resolve()
    own_report = reporter is None
    reporter = reporter or RunReport(root, "condition_C_validation_pilot")
    bundle = root / "Datasets/PHM15-diagnosis/validation_pilot_v1"
    cache = root / "Outputs/diagnosis/C/responses"
    requests, records = [], []
    try:
        config = reporter.configure(root / "Files/experiment_config.json")
        with reporter.step("prepare_C_requests"):
            contract, records, requests = prepare_requests(bundle, config, max_cases)
            atomic_json(reporter.folder / "export_contract_snapshot.json", contract)
            atomic_text(reporter.folder / "diagnostic_requests.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests))
            for name in ("phm_diagnosis.py", "phm_diagnosis_qwen.py", "phm_run_reporting.py", "requirements-qwen-colab.txt"):
                reporter.add_artifact(root / "Files" / name)
            reporter.note("C-only validation pilot. D and the final held-out evaluation have not been executed. No literature is loaded.")
            capture_diagnostics(reporter, bundle, cache, requests, records)
        with reporter.step("Qwen_C_inference"):
            result = _infer(requests, records, cache, config, reporter, retry_errors)
        with reporter.step("score_C_validation_pilot"):
            result["pilot_metrics"] = capture_diagnostics(reporter, bundle, cache, requests, records, score=True)
        return result
    finally:
        # Errors and progress have already been recorded by step(). Preserve the original exception.
        if requests and "diagnostics" not in reporter.data["metrics"]:
            try:
                capture_diagnostics(reporter, bundle, cache, requests, records)
            except Exception as exc:
                reporter.note(f"Could not inspect responses: {type(exc).__name__}: {exc}")
        if own_report:
            reporter.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_condition_C(args.project, args.max_cases, args.retry_errors), indent=2))
