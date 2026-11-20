"""Export validation evidence locally from the existing fitted PHM case system.

Run in the original numerical environment, not Colab. No fitting or final-test
outcome access occurs. Colab consumes JSON, avoiding pickled-model compatibility.
"""

from collections import Counter
from pathlib import Path
from time import perf_counter
import argparse
import json

from phm_diagnosis import LABELS, fingerprint, validate_evidence
from phm_run_reporting import RunReport, atomic_json, atomic_text, load_experiment_config


def select_pilot(frame, count, seed):
    """Round-robin classes; prefer unused sites, then a seeded case-ID hash.

    Validation labels are used for balanced development sampling only. They are
    never passed into the numerical predictor, retriever or language-model input.
    """
    if type(count) is not int or not 3 <= count <= len(frame):
        raise ValueError("Select at least three validation cases within the available cohort.")
    used_ids, used_sites, selected = set(), set(), []
    for index in range(count):
        label = LABELS[index % len(LABELS)]
        choices = frame.loc[frame.reference_label.eq(label) & ~frame.case_id.isin(used_ids)]
        if choices.empty:
            raise ValueError("Not enough examples for the declared class-balanced pilot.")
        records = list(choices.itertuples())
        best = min(records, key=lambda r: (r.site_id in used_sites, fingerprint([seed, str(r.case_id)])))
        selected.append(best.Index)
        used_ids.add(best.case_id)
        used_sites.add(best.site_id)
    return frame.loc[selected].copy()


def export_pilot(project_root, output_dir=None, reporter=None):
    import duckdb
    from phm_case_system import load_system

    root = Path(project_root).resolve()
    output = Path(output_dir) if output_dir else root / "Datasets/PHM15-diagnosis/validation_pilot_v1"
    config = load_experiment_config(root / "Files/experiment_config.json")
    source = root / "Datasets/PHM15-prepared/case_system.duckdb"
    with duckdb.connect(str(source), read_only=True) as db:
        validation = db.sql("select * from private_validation_cases order by case_id").df()
        case_contract = json.loads(db.sql("select contract_json from case_system_contract").fetchall()[0][0])
        schema = db.sql("select * from feature_schema").df().to_dict("records")
        payload_sha = db.sql("select sha256 from fitted_system").fetchall()[0][0]
    # Verify the original numerical code before deserialising the trusted local model.
    for name, expected in case_contract["code_sha256"].items():
        if fingerprint((root / "Files" / name).read_bytes()) != expected:
            raise ValueError(f"Numerical source changed since fitting: {name}. Review before exporting.")
    bank = load_system(source)
    with duckdb.connect(str(root / "Datasets/PHM15-prepared/case_inputs.duckdb"), read_only=True) as db:
        splits = dict(db.sql("select site_id,split from site_split").fetchall())
    if any(splits[int(site)] != "train" for site in bank.training.site_id):
        raise ValueError("Historical bank contains a non-training site.")
    if any(splits[int(site)] != "validation" for site in validation.site_id):
        raise ValueError("Pilot cohort contains a non-validation site.")
    if set(bank.training.case_id) & set(validation.case_id) or set(bank.training.site_id) & set(validation.site_id):
        raise ValueError("Training and validation cases/sites overlap.")
    selected = select_pilot(validation, config["evaluation"]["validation_pilot_queries"], config["generator"]["seed"])
    order = list(bank.features)
    if {s["feature"] for s in schema} != set(order):
        raise ValueError("Saved feature schema differs from fitted retrieval features.")
    dictionary = {
        "feature_order": order, "retrieval_schema": schema,
        "notes": [
            "Channel identifiers are anonymised: do not assign physical meanings or units.",
            "Window: 3 hours on a 15-minute grid. Summaries aggregate component-level measurements.",
            "IQR is the interquartile range; component changes are last minus first observed values.",
            "Recent summaries use the last hour; early-to-recent change compares first-hour and last-hour component medians.",
            "Slopes use observed times in hours. ch3 transition rate uses adjacent observed bins; longest runs are divided by 12.",
            "Coverage and occupancy fractions are proportions, not fault probabilities.",
            "Distances use group-balanced, training-standardised prepared features; smaller is more similar, with no universal cutoff."
        ]}
    evidence, labels, audit = [], [], []
    private_columns = ["site_id", "decision_time", "episode_id", "reference_label", "window_start", "window_end", "first_observation", "last_observation"]
    training_lookup = bank.training.set_index("case_id")
    for _, row in selected.iterrows():
        query = row.to_frame().T.drop(columns=private_columns)
        start = perf_counter()
        record = bank.evidence(query, site_id=int(row.site_id), k=3)
        seconds = perf_counter() - start
        validate_evidence(record, order)
        neighbour_ids = [item["case_id"] for item in record["historical_cases"]]
        neighbour_sites = [int(training_lookup.loc[key, "site_id"]) for key in neighbour_ids]
        if len(set(neighbour_sites)) != 3 or int(row.site_id) in neighbour_sites:
            raise ValueError("Historical cases violate distinct-site or query-site exclusions.")
        evidence.append(record)
        labels.append({"case_id": record["case_id"], "reference_label": str(row.reference_label)})
        audit.append({"case_id": record["case_id"], "site_id": int(row.site_id), "split": "validation",
                      "historical_case_ids": neighbour_ids, "historical_site_ids": neighbour_sites,
                      "evidence_construction_seconds": seconds})
    files = {
        "validation_evidence.jsonl": "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in evidence),
        "private_validation_labels.jsonl": "".join(json.dumps(row) + "\n" for row in labels),
        "feature_dictionary.json": json.dumps(dictionary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    }
    contract = {
        "schema_version": "0.7.0", "scope": "validation_pilot", "historical_k": 3,
        "model_id": bank.model_id, "case_system_sha256": fingerprint(source.read_bytes()),
        "fitted_payload_sha256": payload_sha, "numerical_code_sha256": case_contract["code_sha256"],
        "case_ids": [r["case_id"] for r in evidence], "native_features": len(order),
        "selection": "class round-robin, prefer unused validation sites, seed plus case-ID SHA256 tie-break",
        "seed": config["generator"]["seed"], "validation_labels_used_only_for_selection_and_scoring": True,
        "final_test_outcomes_accessed": False, "query_private_fields_excluded": private_columns,
        "class_counts": dict(Counter(r["reference_label"] for r in labels)),
        "validation_sites": len(set(selected.site_id)), "training_cases": len(bank.training),
        "files": {name: fingerprint(content.encode("utf-8")) for name, content in files.items()}}
    manifest = output / "export_contract.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != contract or any(not (output / name).exists() or fingerprint((output / name).read_bytes()) != expected
                                       for name, expected in contract["files"].items()):
            raise ValueError("Existing export differs. Use a new versioned output folder; do not replace pilot inputs silently.")
    else:
        for name, content in files.items():
            atomic_text(output / name, content)
        atomic_json(output / "private_export_audit.json", audit)
        atomic_json(manifest, contract)
    if reporter:
        reporter.data["metrics"]["case_export"] = {"queries": len(evidence), "features": len(order), "historical_k": 3,
                                                    "validation_sites": contract["validation_sites"], "class_counts": contract["class_counts"],
                                                    "final_test_outcomes_accessed": False}
        reporter.data["remaining_pipeline"] = ["run condition C validation pilot in Colab", "review permitted literature for D",
                                              "paired C/D validation and frozen settings", "held-out paired C/D evaluation"]
        for name in [*files, "export_contract.json", "private_export_audit.json"]:
            reporter.add_artifact(output / name)
        reporter.save()
    return output, contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = RunReport(args.project, "condition_C_evidence_export")
    try:
        report.configure(args.project / "Files/experiment_config.json")
        with report.step("export_validation_evidence"):
            folder, contract = export_pilot(args.project, args.output, report)
            print(f"Exported {len(contract['case_ids'])} validation cases: {folder}")
        report.note("Local evidence export only. No Qwen diagnostic inference and no final test scoring performed.")
    finally:
        report.finish()
