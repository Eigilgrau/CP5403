"""Run resumable Qwen boundary selection on a CUDA Colab runtime."""

from pathlib import Path
from time import perf_counter
import argparse
import importlib.metadata
import json
import platform

from phm_literature import QWEN, QWEN_REVISION, SETTINGS, atomic_json, read_json, check_cached_source_policy


def _run_qwen(output_dir, max_requests=None, retry_errors=False, reporter=None):
    """Cache each actual model response before continuing to the next request."""
    check_cached_source_policy(output_dir)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the GPU section in Colab; no inference was performed.")
    if max_requests is not None and (not isinstance(max_requests, int) or max_requests < 1):
        raise ValueError("max_requests must be a positive integer or None.")
    folder = Path(output_dir)
    requests = [json.loads(line) for line in (folder / "qwen_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    pending = []
    for request in requests:
        cache = folder / "qwen_responses" / (request["request_id"] + ".json")
        if not cache.exists() or (retry_errors and read_json(cache).get("error")):
            pending.append(request)
    if max_requests is not None:
        pending = pending[:max_requests]
    if not pending:
        return {"new_responses": 0, "reason": "all requested batches already cached"}
    set_seed(SETTINGS["seed"])
    started = perf_counter()
    if reporter:
        reporter.event("model_loading", model=QWEN, revision=QWEN_REVISION)
        reporter.data["metrics"]["qwen_progress"] = {"status": "loading_model", "requested_batches": len(pending)}
        reporter.save()
    tokenizer = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION)
    quantisation = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(QWEN, revision=QWEN_REVISION,
        quantization_config=quantisation, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    runtime = {"model": QWEN, "revision": QWEN_REVISION, "python": platform.python_version(),
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
               "packages": {name: importlib.metadata.version(name) for name in
                            ["torch", "transformers", "accelerate", "bitsandbytes", "tokenizers"]},
               "quantisation": "NF4 4-bit, double quantisation, float16 compute",
               "model_load_seconds": perf_counter() - started, "settings": SETTINGS}
    atomic_json(folder / "qwen_runtime.json", runtime)
    if reporter:
        reporter.data["environment"]["qwen_runtime"] = runtime
        reporter.refresh_packages()
        reporter.save()
    for request in pending:
        if request["model"] != QWEN or request["revision"] != QWEN_REVISION or request["settings"] != SETTINGS:
            raise ValueError("Request and runner contracts differ; regenerate requests first.")
        started = perf_counter()
        if reporter:
            reporter.event("qwen_request_started", request_id=request["request_id"])
            reporter.data["metrics"]["qwen_progress"] = {"status": "generating", "request_id": request["request_id"]}
            reporter.save()
        result = {"origin": "model_inference", "request_id": request["request_id"],
                  "model": QWEN, "revision": QWEN_REVISION, "runtime": runtime,
                  "raw_output": None, "error": None}
        try:
            inputs = tokenizer.apply_chat_template(request["messages"], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            result["input_tokens"] = int(inputs["input_ids"].shape[-1])
            if result["input_tokens"] > SETTINGS["max_input_tokens"]:
                raise ValueError("Prompt exceeds the declared local inference budget; no truncation applied.")
            inputs = inputs.to(model.device)
            torch.cuda.synchronize()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False,
                    max_new_tokens=SETTINGS["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tail = generated[0, inputs["input_ids"].shape[-1]:]
            result["output_tokens"] = len(tail)
            result["raw_output"] = tokenizer.decode(tail, skip_special_tokens=True)
            result["generation_hit_limit"] = len(tail) >= SETTINGS["max_new_tokens"]
            del inputs, generated, tail
        except (RuntimeError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        result["seconds"] = perf_counter() - started
        atomic_json(folder / "qwen_responses" / (request["request_id"] + ".json"), result)
        if reporter:
            reporter.event("qwen_response_cached", request_id=request["request_id"], error=result["error"],
                           seconds=result["seconds"], input_tokens=result.get("input_tokens"), output_tokens=result.get("output_tokens"))
            reporter.data["metrics"]["qwen_progress"] = {"status": "response_cached", "request_id": request["request_id"],
                                                        "error": result["error"]}
            reporter.capture_literature(folder)
        print(request["request_id"][:12], result["error"] or "cached", flush=True)
        if result["error"] and "out of memory" in result["error"].lower():
            break
    return {"requested_batches": len(pending), "runtime": runtime}


def run_qwen(output_dir, max_requests=None, retry_errors=False, reporter=None):
    """Run inference with persistent setup errors, progress, and cache summaries."""
    from phm_run_reporting import RunReport
    own_report = reporter is None
    if own_report:
        reporter = RunReport(Path(__file__).resolve().parent.parent, "qwen_chunking")
    try:
        if own_report:
            reporter.configure(Path(__file__).with_name("experiment_config.json"))
        with reporter.step("qwen_boundary_inference"):
            result = _run_qwen(output_dir, max_requests, retry_errors, reporter)
        return result
    finally:
        # Preserve the original failure if inspecting a damaged cache also fails.
        try:
            reporter.capture_literature(output_dir)
        except Exception as exc:
            reporter.note(f"Could not snapshot the literature cache: {type(exc).__name__}: {exc}")
        if own_report:
            reporter.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_qwen(args.output, args.max_requests, args.retry_errors), indent=2))
