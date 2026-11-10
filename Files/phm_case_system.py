"""Represent PHM cases, validate a classifier, and retrieve historical evidence."""

from collections import Counter
from io import BytesIO
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5
import hashlib
import json
import warnings

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from phm_case_design import FoldTransform, MODEL_FEATURES, LABELS, save_frame
from phm_case_samples import summarise_window
from phm_preprocessing import MIN_BINS, LOOKBACK, STEP, SEED


CONTINUOUS_CHANNELS = ("ch1", "ch2", "ch5")
CODE_CHANNELS = ("ch3", "ch6", "ch7")
TEMPORAL_FEATURES = [
    f"{ch}_{stat}" for ch in CONTINUOUS_CHANNELS
    for stat in ("recent_component_median", "recent_lowest_component_median",
                 "recent_highest_component_median", "early_to_recent_change",
                 "median_component_slope_per_hour")
]
TEMPORAL_FEATURES += ["recent_usable_component_fraction", "ch3_mean_transition_rate", "ch3_mean_longest_run_fraction"]
CANDIDATES = (
    ("summary_linear", False, False, False),
    ("summary_spline", False, False, True),
    ("temporal_linear", True, False, False),
    ("temporal_spline", True, False, True),
    ("temporal_codes_linear", True, True, False),
    ("temporal_codes_spline", True, True, True),
)
PROTOCOL = {
    "version": "0.3.0", "seed": SEED, "training_cases": "retained v0.2 cohort",
    "validation_site_sampling": "same seeded one-episode, three-hour spacing and 30-per-class cap",
    "coverage_threshold": 0.75, "candidate_names": [c[0] for c in CANDIDATES],
    "regularisation_C": 1.0, "class_weight": "balanced", "clip_quantiles": [0.005, 0.995],
    "spline_knots": 4, "spline_degree": 3, "category_limit_per_channel": 5,
    "selection": "highest validation macro-F1; within 0.005 prefer fewer coefficients, then lower log loss",
    "retrieval": "three nearest training cases, distinct sites, group-balanced standardised base features",
    "test_usage": "reserved for the frozen evaluation"
}


def attach_views(db, source):
    path = str(Path(source).resolve()).replace("'", "''")
    db.execute(f"attach '{path}' as source (read_only)")
    for split in ("train", "validation"):
        for short, table in (("bins", "component_bins"), ("decisions", "private_decisions"),
                             ("inventory", "site_inventory")):
            db.execute(f"""create temp view {split}_{short} as
                select t.* from source.{table} t join source.site_split s using(site_id)
                where s.split='{split}'""")


def extended_window(window, time):
    """Describe recent movement and code occupancy using observations available by t."""
    ordered = window.sort_values(["component", "bin_end"])
    indices = list(ordered.groupby("component", sort=False).indices.values())
    names = (*CONTINUOUS_CHANNELS, *CODE_CHANNELS)
    values = ordered[list(names)].to_numpy(dtype=float)
    hours = (ordered.observed_at - time).dt.total_seconds().to_numpy() / 3600
    bins = (ordered.bin_end - time).dt.total_seconds().to_numpy() / 900
    summaries = {ch: {"recent": [], "change": [], "slope": []} for ch in CONTINUOUS_CHANNELS}
    profiles = {ch: [] for ch in CODE_CHANNELS}
    transitions, runs = [], []
    for at in indices:
        component_values, component_hours, component_bins = values[at], hours[at], bins[at]
        for j, ch in enumerate(CONTINUOUS_CHANNELS):
            valid = np.isfinite(component_values[:, j])
            if valid.sum() < MIN_BINS:
                continue
            y, x, clock = component_values[valid, j], component_hours[valid], component_bins[valid]
            early, late = y[clock <= -8], y[clock > -4]
            if len(late) >= 2:
                summaries[ch]["recent"].append(float(np.median(late)))
            if len(early) >= 2 and len(late) >= 2:
                summaries[ch]["change"].append(float(np.median(late) - np.median(early)))
            centered = x - x.mean()
            summaries[ch]["slope"].append(float(np.dot(centered, y) / np.dot(centered, centered)))
        for j, ch in enumerate(CODE_CHANNELS, start=len(CONTINUOUS_CHANNELS)):
            valid = np.isfinite(component_values[:, j])
            if valid.sum() < MIN_BINS:
                continue
            codes, counts = np.unique(component_values[valid, j], return_counts=True)
            profiles[ch].append({format(float(k), ".17g"): float(n / valid.sum())
                                 for k, n in zip(codes, counts)})
            if ch == "ch3":
                code_values, clock = component_values[valid, j], component_bins[valid]
                contiguous = np.diff(clock) == 1
                same = np.diff(code_values) == 0
                if contiguous.sum():
                    transitions.append(float((contiguous & ~same).sum() / contiguous.sum()))
                longest = current = 1
                for adjacent, unchanged in zip(contiguous, same):
                    current = current + 1 if adjacent and unchanged else 1
                    longest = max(longest, current)
                runs.append(longest / 12)
    result = {}
    for ch, summary in summaries.items():
        recent = summary["recent"]
        result[f"{ch}_recent_component_median"] = float(np.median(recent)) if recent else np.nan
        result[f"{ch}_recent_lowest_component_median"] = min(recent) if recent else np.nan
        result[f"{ch}_recent_highest_component_median"] = max(recent) if recent else np.nan
        result[f"{ch}_early_to_recent_change"] = float(np.median(summary["change"])) if summary["change"] else np.nan
        result[f"{ch}_median_component_slope_per_hour"] = float(np.median(summary["slope"])) if summary["slope"] else np.nan
    result["recent_usable_component_fraction"] = len(summaries["ch1"]["recent"]) / len(indices)
    for ch, channel_profiles in profiles.items():
        total = Counter()
        for profile in channel_profiles:
            total.update(profile)
        result[f"{ch}_profile"] = {k: v / len(channel_profiles) for k, v in sorted(total.items())} if channel_profiles else {}
    result["ch3_mean_transition_rate"] = float(np.mean(transitions)) if transitions else np.nan
    result["ch3_mean_longest_run_fraction"] = float(np.mean(runs)) if runs else np.nan
    return result


def augment_points(db, points, split):
    if split not in {"train", "validation"}:
        raise ValueError("Choose a development split.")
    records = []
    for site, selected in points.groupby("site_id"):
        db.register("window_points", selected[["case_id", "site_id", "decision_time"]])
        rows = db.execute(f"""select p.case_id,b.* from {split}_bins b join window_points p
            on b.site_id=p.site_id and b.bin_end>p.decision_time-interval '3 hours'
            and b.bin_end<=p.decision_time where b.site_id=?
            order by p.case_id,b.component,b.bin_end""", [int(site)]).df()
        db.unregister("window_points")
        times = selected.set_index("case_id").decision_time
        for case_id, window in rows.groupby("case_id", sort=False):
            time = times[case_id]
            assert window.observed_at.between(time - LOOKBACK, time, inclusive="right").all()
            records.append({"case_id": case_id, **extended_window(window, time)})
        if len(records) % 300 < len(selected):
            print(f"Extended {split} features for {len(records)} cases", flush=True)
    return pd.DataFrame(records)


def save_features(db, name, frame):
    stored = frame.copy()
    for ch in CODE_CHANNELS:
        stored[f"{ch}_profile"] = stored[f"{ch}_profile"].map(lambda v: json.dumps(v, sort_keys=True))
    save_frame(db, name, stored)


def read_features(db, name):
    frame = db.execute(f"select * from {name} order by case_id").df()
    for ch in CODE_CHANNELS:
        frame[f"{ch}_profile"] = frame[f"{ch}_profile"].map(json.loads)
    return frame


class CaseEncoder(TransformerMixin, BaseEstimator):
    """Keep native summary features and learn category proportions without code ordering."""

    def __init__(self, temporal=False, codes=False):
        self.temporal = temporal
        self.codes = codes

    def fit(self, X, y=None):
        self.numeric_ = MODEL_FEATURES + (TEMPORAL_FEATURES if self.temporal else [])
        self.vocabulary_ = {}
        if self.codes:
            for ch in CODE_CHANNELS:
                total = Counter()
                for profile in X[f"{ch}_profile"]:
                    total.update(profile)
                self.vocabulary_[ch] = [k for k, _ in sorted(total.items(), key=lambda p: (-p[1], p[0]))[:5]]
        self.feature_names_out_ = self.transform(X.iloc[:1]).columns.to_numpy()
        return self

    def transform(self, X):
        output = X[self.numeric_].astype(float).copy()
        for ch, levels in self.vocabulary_.items():
            profiles = X[f"{ch}_profile"]
            # omit the most frequent code as the reference category
            for code in levels[1:]:
                output[f"{ch}_code_{code}_fraction"] = profiles.map(lambda p: p.get(code, 0.0))
            output[f"{ch}_other_code_fraction"] = profiles.map(
                lambda p: sum(value for code, value in p.items() if code not in levels))
        return output

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class CurvedFeatures(TransformerMixin, BaseEstimator):
    """Add curved effects to numeric levels and trends while retaining proportion terms."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.curved_ = [c for c in X.columns if "_fraction" not in c and c != "bin_coverage"
                        and "transition_rate" not in c and X[c].nunique() >= 6]
        self.straight_ = [c for c in X.columns if c not in self.curved_]
        self.spline_ = SplineTransformer(n_knots=4, degree=3, include_bias=False,
                                        knots="quantile", extrapolation="constant")
        if self.curved_:
            self.spline_.fit(X[self.curved_])
        return self

    def transform(self, X):
        pieces = [X[self.straight_].to_numpy()]
        if self.curved_:
            pieces.insert(0, self.spline_.transform(X[self.curved_]))
        return np.column_stack(pieces)

    def get_feature_names_out(self, input_features=None):
        curved = self.spline_.get_feature_names_out(self.curved_) if self.curved_ else []
        return np.asarray([*curved, *self.straight_])


class NamedPreparation(FoldTransform):
    """Retain names after the existing imputation and redundancy screen."""

    def transform(self, X):
        values = super().transform(X)
        return pd.DataFrame(values, index=X.index, columns=self.get_feature_names_out())


def make_model(name):
    spec = next(c for c in CANDIDATES if c[0] == name)
    _, temporal, codes, curved = spec
    steps = [("encode", CaseEncoder(temporal, codes)), ("prepare", NamedPreparation(clip=True))]
    if curved:
        steps.append(("basis", CurvedFeatures()))
    steps += [("scale", StandardScaler()),
              ("logistic", LogisticRegression(C=1.0, l1_ratio=0.0, class_weight="balanced",
                                              solver="lbfgs", max_iter=2000))]
    return Pipeline(steps)


def fit_model(model, X, y):
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("A candidate failed to converge within the declared iteration limit.")
    model.training_case_ids_ = tuple(X.case_id)
    model.training_site_ids_ = tuple(sorted(X.site_id.unique()))
    return model


def compare_representations(train, validation):
    rows, scored, models = [], [], {}
    for name, *_ in CANDIDATES:
        started = perf_counter()
        model = fit_model(make_model(name), train, train.reference_label)
        probabilities = model.predict_proba(validation)
        classes = model.named_steps["logistic"].classes_
        predicted = classes[probabilities.argmax(axis=1)]
        report = classification_report(validation.reference_label, predicted, labels=LABELS,
                                       output_dict=True, zero_division=0)
        rows.append((name, f1_score(validation.reference_label, predicted, average="macro"),
                     log_loss(validation.reference_label, probabilities, labels=classes),
                     report[LABELS[1]]["recall"], report[LABELS[2]]["recall"],
                     model.named_steps["logistic"].coef_.shape[1],
                     int(model.named_steps["logistic"].n_iter_.max()), perf_counter() - started))
        scores = pd.DataFrame(probabilities, columns=[f"score_{c}" for c in classes])
        scores.insert(0, "case_id", validation.case_id.to_numpy())
        scores["candidate"] = name
        scores["prediction"] = predicted
        scored.append(scores)
        models[name] = model
        print(f"Validated {name}", flush=True)
    metrics = pd.DataFrame(rows, columns=["candidate", "macro_f1", "log_loss", "runtime_recall",
        "supply_recall", "coefficients_per_class", "iterations", "fit_and_score_seconds"])
    close = metrics[metrics.macro_f1 >= metrics.macro_f1.max() - 0.005]
    winner = close.sort_values(["coefficients_per_class", "log_loss", "candidate"]).iloc[0].candidate
    return metrics, pd.concat(scored, ignore_index=True), winner, models[winner]


def cross_fitted_scores(train, candidate):
    outputs, memberships = [], []
    for fold, (fit, check) in enumerate(GroupKFold(5).split(train, groups=train.site_id)):
        assert not set(train.iloc[fit].site_id) & set(train.iloc[check].site_id)
        model = fit_model(make_model(candidate), train.iloc[fit], train.iloc[fit].reference_label)
        classes = model.named_steps["logistic"].classes_
        probability = model.predict_proba(train.iloc[check])
        frame = pd.DataFrame(probability, columns=[f"score_{c}" for c in classes])
        frame.insert(0, "case_id", train.iloc[check].case_id.to_numpy())
        frame["model_id"] = f"v0.3-{candidate}-fold{fold}"
        frame["fold"] = fold
        outputs.append(frame)
        memberships.extend((fold, int(s), "fit") for s in train.iloc[fit].site_id.unique())
        memberships.extend((fold, int(s), "check") for s in train.iloc[check].site_id.unique())
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(memberships, columns=["fold", "site_id", "role"])


def feature_group(name):
    if name.startswith("ch"):
        return name.split("_", 1)[0]
    if name in {"occupied_fraction", "setback_fraction"}:
        return "operating_state"
    return "coverage"


class CaseSystem:
    """Serve numerical predictions and traceable neighbours from a frozen training bank."""

    def __init__(self, model, candidate, train, oof_scores):
        if set(train.case_id) != set(model.training_case_ids_):
            raise ValueError("The historical bank must match the model's training cases.")
        if set(oof_scores.case_id) != set(train.case_id) or not oof_scores.case_id.is_unique:
            raise ValueError("Each historical case needs exactly one out-of-fold score.")
        self.model = model
        self.candidate = candidate
        self.model_id = f"phm2015-v0.3-{candidate}"
        self.classes = model.named_steps["logistic"].classes_.tolist()
        self.training = train.reset_index(drop=True).copy()
        self.oof_scores = oof_scores.set_index("case_id")
        base = model[:2].transform(self.training)
        self.features = list(base.columns)
        self.search_scale = StandardScaler().fit(base)
        groups = [feature_group(c) for c in self.features]
        counts = Counter(groups)
        self.weights = np.array([1 / np.sqrt(counts[g] * len(counts)) for g in groups])
        self.vectors = self.search_scale.transform(base) * self.weights
        self.encoded_training = model.named_steps["encode"].transform(self.training)

    def retrieve(self, query, site_id=None, k=3):
        if len(query) != 1 or k < 1:
            raise ValueError("Supply one query and a positive neighbour count.")
        base = self.model[:2].transform(query)
        vector = self.search_scale.transform(base)[0] * self.weights
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        eligible = self.training.case_id.ne(query.iloc[0].case_id).to_numpy(copy=True)
        known = self.training.loc[self.training.case_id == query.iloc[0].case_id, "site_id"]
        if site_id is None and len(known):
            site_id = int(known.iloc[0])
        if site_id is not None:
            eligible &= self.training.site_id.ne(site_id).to_numpy()
        distances = np.where(eligible, distances, np.inf)
        order = np.argsort(distances, kind="stable")
        rows, seen_sites = [], set()
        for index in order:
            if not np.isfinite(distances[index]):
                break
            historical = self.training.iloc[index]
            if historical.site_id in seen_sites:
                continue
            seen_sites.add(historical.site_id)
            rows.append((int(index), float(distances[index])))
            if len(rows) == k:
                break
        return rows

    def evidence(self, query, site_id=None, k=3):
        encoded = self.model.named_steps["encode"].transform(query)
        scores = self.model.predict_proba(query)[0]
        native = encoded.iloc[0][self.features]
        prepare = self.model.named_steps["prepare"]
        clipped, imputed = [], []
        for i in prepare.selected_:
            value, name = encoded.iloc[0, i], encoded.columns[i]
            if pd.isna(value):
                imputed.append(name)
            elif value < prepare.lower_[i] or value > prepare.upper_[i]:
                clipped.append(name)
        record = {"case_id": str(query.iloc[0].case_id), "lookback_minutes": 180,
                  "feature_values": json_values(native), "model_id": self.model_id,
                  "preparation_flags": {"clipped_features": clipped, "imputed_features": imputed},
                  "quality": json_values(query.iloc[0][[
                      "observed_components", "usable_components_ch1", "observed_component_bins_ch1",
                      "possible_component_bins_ch1", "usable_component_fraction",
                      "recent_component_fraction", "bin_coverage"]]),
                  "class_scores": dict(zip(self.classes, map(float, scores))), "historical_cases": []}
        raw_query = self.search_scale.transform(self.model[:2].transform(query))[0]
        for index, distance in self.retrieve(query, site_id, k):
            other = self.training.iloc[index]
            raw_other = self.vectors[index] / self.weights
            difference = np.abs(raw_query - raw_other)
            informative = np.array([not c.endswith("_fraction") and feature_group(c).startswith("ch")
                                    for c in self.features])
            possibilities = np.flatnonzero(informative)
            nearest = possibilities[np.argmin(difference[possibilities])] if len(possibilities) else int(np.argmin(difference))
            furthest = int(np.argmax(difference))
            comparison = []
            for role, at in (("closest_numeric_feature", nearest), ("largest_standardised_difference", furthest)):
                feature = self.features[at]
                comparison.append({"comparison": role, "feature": feature,
                    "query_value": json_number(native[feature]),
                    "historical_value": json_number(self.encoded_training.iloc[index][feature]),
                    "absolute_prepared_standardised_gap": float(difference[at])})
            record["historical_cases"].append({
                "case_id": str(other.case_id), "recorded_outcome": str(other.reference_label),
                "distance": distance, "feature_values": json_values(self.encoded_training.iloc[index][self.features]),
                "comparison": comparison})
        return record


def json_number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def json_values(series):
    return {str(k): json_number(v) for k, v in series.items()}


def persist_system(db, system):
    stream = BytesIO()
    joblib.dump(system, stream, compress=3)
    payload = stream.getvalue()
    db.execute("create table fitted_system (model_id varchar, sha256 varchar, payload blob)")
    db.execute("insert into fitted_system values (?,?,?)",
               [system.model_id, hashlib.sha256(payload).hexdigest(), payload])


def load_system(path):
    with duckdb.connect(str(path), read_only=True) as db:
        _, digest, payload = db.execute("select * from fitted_system").fetchall()[0]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("The stored model payload failed its integrity check.")
    return joblib.load(BytesIO(payload))


def query_from_measurements(db, site_id, time):
    """Create inference features using only the component table and timestamps."""
    time = pd.Timestamp(time)
    if pd.isna(time) or time.tzinfo is not None or time != time.floor(STEP):
        raise ValueError("Use a timezone-naive quarter-hour time on the dataset clock.")
    first = db.execute("select min(observed_at) from component_bins where site_id=?", [site_id]).fetchall()[0][0]
    if first is None or time - LOOKBACK < pd.Timestamp(first):
        return None, "Insufficient three-hour history."
    rows = db.execute("""select * from component_bins where site_id=? and bin_end>?
        and bin_end<=? order by component,bin_end""", [site_id, time - LOOKBACK, time]).df()
    if rows.empty:
        return None, "No observed component bins in the window."
    assert rows.observed_at.between(time - LOOKBACK, time, inclusive="right").all()
    recent = db.execute("""select count(distinct component) from component_bins
        where site_id=? and bin_end>? and bin_end<=?""",
                        [site_id, time - pd.Timedelta(hours=24), time]).fetchall()[0][0]
    features = summarise_window(rows)
    count = features["observed_components"]
    features.update(usable_component_fraction=features["usable_components_ch1"] / count,
                    recent_component_fraction=count / recent,
                    bin_coverage=features["observed_component_bins_ch1"] / (12 * count),
                    occupied_fraction=rows.operating_state.eq("Occupied").mean(),
                    setback_fraction=rows.operating_state.eq("Setback").mean())
    if features["usable_component_fraction"] < 0.75 or features["recent_component_fraction"] < 0.75:
        return None, "Component coverage is below the declared 0.75 threshold."
    features.update(extended_window(rows, time))
    case_id = uuid5(NAMESPACE_URL, f"phm2015/case/v1/{site_id}/{time.isoformat()}").hex
    return pd.DataFrame([{"case_id": case_id, **features}]), None


def retrieval_audit(system, validation):
    links, diagnostics = [], []
    for i, row in enumerate(validation.itertuples()):
        started = perf_counter()
        neighbours = system.retrieve(validation.iloc[[i]], site_id=row.site_id)
        elapsed = perf_counter() - started
        if len(neighbours) != 3:
            raise ValueError("The bank needs at least three other training sites.")
        outcomes = [system.training.iloc[index].reference_label for index, _ in neighbours]
        counts = Counter(outcomes)
        vote = next(label for label in outcomes if counts[label] == max(counts.values()))
        diagnostics.append((row.case_id, row.reference_label, row.reference_label in outcomes, vote, elapsed))
        links.extend((row.case_id, rank, system.training.iloc[index].case_id, distance)
                     for rank, (index, distance) in enumerate(neighbours, 1))
    return (pd.DataFrame(links, columns=["query_case_id", "rank", "historical_case_id", "distance"]),
            pd.DataFrame(diagnostics, columns=["case_id", "reference_label", "hit_at_3", "neighbour_vote", "seconds"]))


def validate_system(db, bank, validation, links, oof, folds):
    train = bank.training
    sites = db.execute("select site_id,split from source.site_split").df()
    assert set(train.site_id) == set(sites.loc[sites.split == "train", "site_id"])
    assert set(validation.site_id) == set(sites.loc[sites.split == "validation", "site_id"])
    assert not set(train.case_id) & set(validation.case_id)
    for split, cases in (("train", train), ("validation", validation)):
        assert cases.case_id.is_unique and cases.episode_id.dropna().is_unique
        ordered = cases.sort_values(["site_id", "decision_time"])
        assert ordered.groupby("site_id").decision_time.diff().dropna().ge(LOOKBACK).all()
        assert (cases.first_observation > cases.window_start).all()
        assert (cases.window_end == cases.decision_time).all()
        assert (cases.last_observation <= cases.window_end).all()
        db.register("cases_to_check", cases[["site_id", "decision_time", "reference_label", "episode_id"]])
        assert db.execute(f"""select count(*) from cases_to_check c left join {split}_decisions d
            on c.site_id=d.site_id and c.decision_time=d.decision_time
            where d.status is distinct from 'eligible' or c.reference_label is distinct from d.reference_label
            or c.episode_id is distinct from d.episode_id""").fetchall() == [(0,)]
        db.unregister("cases_to_check")
    assert links.groupby("query_case_id").size().eq(3).all()
    assert set(links.query_case_id) == set(validation.case_id)
    joined = links.merge(train[["case_id", "site_id"]], left_on="historical_case_id", right_on="case_id", validate="many_to_one")
    assert len(joined) == len(links)
    assert joined.groupby("query_case_id").site_id.nunique().eq(3).all()
    assert not (links.query_case_id == links.historical_case_id).any()
    historical_query = train.iloc[[0]]
    all_neighbours = bank.retrieve(historical_query, k=len(train))
    assert len(all_neighbours) == train.site_id.nunique() - 1
    assert all(train.iloc[index].site_id != historical_query.iloc[0].site_id
               for index, _ in all_neighbours)
    assert oof.case_id.is_unique and set(oof.case_id) == set(train.case_id)
    for _, group in folds.groupby("fold"):
        assert not set(group.loc[group.role == "fit", "site_id"]) & set(group.loc[group.role == "check", "site_id"])
    probability = bank.model.predict_proba(validation)
    assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
    changed = validation.copy()
    changed["reference_label"] = "deliberately altered private label"
    assert np.array_equal(probability, bank.model.predict_proba(changed))
    assert bank.retrieve(validation.iloc[[0]]) == bank.retrieve(changed.iloc[[0]])
    for ch in CODE_CHANNELS:
        assert validation[f"{ch}_profile"].map(lambda p: not p or abs(sum(p.values()) - 1) < 1e-10).all()
    return ["Training and Validation Site Isolation", "Unique Episodes and Non-Overlapping Cases",
            "Actual Observation Time Boundaries", "Labels Match the Eligible Decision Table",
            "Three Historical Cases From Distinct Training Sites", "Same-Site Exclusion for Large Retrieval Requests",
            "Complete Grouped Out-of-Fold Scores",
            "Finite Class Scores Sum to One", "Private Query Labels Do Not Affect Prediction or Retrieval",
            "Code Profiles Are Valid Proportions"]
