"""Scout training sites, construct case summaries, and diagnose a logistic pilot."""

from bisect import bisect_left, insort
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from phm_case_samples import summarise_window
from phm_preprocessing import CHANNELS, CONTROL, LOOKBACK, MIN_BINS, SEED, TARGETS


LABELS = (CONTROL, *TARGETS)
CASE_CAP = 30
QUALITY_COLUMNS = ("usable_component_fraction", "recent_component_fraction", "bin_coverage")
CORE_FEATURES = [
    f"{ch}_{stat}" for ch in CHANNELS
    for stat in ("median_of_component_medians", "median_of_component_iqrs",
                 "median_of_component_changes", "lowest_component_median", "highest_component_median")
]
ALL_FEATURES = CORE_FEATURES + list(QUALITY_COLUMNS) + ["occupied_fraction", "setback_fraction"]
MODEL_FEATURES = [name for name in ALL_FEATURES if not name.startswith(("ch3_", "ch6_", "ch7_"))]


def attach_training_views(db, source):
    path = str(source.resolve()).replace("'", "''")
    db.execute(f"attach '{path}' as source (read_only)")
    for short, table in (("bins", "component_bins"), ("decisions", "private_decisions"),
                         ("episodes", "private_episodes"), ("inventory", "site_inventory")):
        db.execute(f"""create temp view train_{short} as select t.* from source.{table} t
            join source.site_split s using(site_id) where s.split='train'""")
    assert db.execute("select count(*) from train_inventory").fetchall()[0][0] == 46


def save_frame(db, name, frame):
    db.register("frame_to_save", frame)
    db.execute(f"create or replace table {name} as select * from frame_to_save")
    db.unregister("frame_to_save")


def channel_audit(db):
    """Compute ranges and approximate quantiles without loading the signal table."""
    fields = []
    for ch in CHANNELS:
        fields.extend([f"count(*)-count({ch})", f"count(distinct {ch})", f"min({ch})",
                       f"approx_quantile({ch}, [0.01,0.25,0.5,0.75,0.99])", f"max({ch})"])
    values = db.execute(f"select {', '.join(fields)} from train_bins").fetchall()[0]
    records = []
    for i, ch in enumerate(CHANNELS):
        missing, unique, low, quantiles, high = values[i * 5:i * 5 + 5]
        records.append((ch, missing, unique, low, *quantiles, high))
    return pd.DataFrame(records, columns=["channel", "nulls", "unique_values", "minimum",
                                         "p01", "p25", "p50", "p75", "p99", "maximum"])


def select_scout_points(db, cap=CASE_CAP, prefix="train"):
    """Give each positive episode one chance and separate all windows within a site."""
    chosen, audit = [], []
    if prefix not in {"train", "validation"}:
        raise ValueError("Use the training or validation views.")
    sites = db.execute(f"select site_id from {prefix}_inventory order by site_id").fetchall()
    for (site,) in sites:
        eligible = db.execute(f"""select * from {prefix}_decisions
            where site_id=? and status='eligible' order by decision_time""", [site]).df()
        rng = np.random.default_rng(SEED + site)
        pools = {}
        for label in LABELS:
            subset = eligible[eligible.reference_label == label]
            if label == CONTROL:
                indices = subset.index.to_numpy()
            else:
                indices = np.array([rng.choice(group.index) for _, group in subset.groupby("episode_id")])
            pools[label] = list(rng.permutation(indices))
        positions = dict.fromkeys(LABELS, 0)
        kept = dict.fromkeys(LABELS, 0)
        overlaps = dict.fromkeys(LABELS, 0)
        accepted = []
        label_order = list(rng.permutation(LABELS))
        while any(kept[label] < cap and positions[label] < len(pools[label]) for label in LABELS):
            for label in label_order:
                if kept[label] >= cap or positions[label] >= len(pools[label]):
                    continue
                row = eligible.loc[pools[label][positions[label]]]
                positions[label] += 1
                time = row.decision_time
                at = bisect_left(accepted, time)
                neighbours = accepted[max(0, at - 1):at + 1]
                if any(abs(time - other) < LOOKBACK for other in neighbours):
                    overlaps[label] += 1
                    continue
                insort(accepted, time)
                kept[label] += 1
                chosen.append({"case_id": uuid5(NAMESPACE_URL, f"phm2015/case/v1/{site}/{time.isoformat()}").hex,
                               "site_id": site, "decision_time": time, "episode_id": row.episode_id,
                               "reference_label": label})
        for label in LABELS:
            subset = eligible[eligible.reference_label == label]
            audit.append((site, label, len(subset), len(pools[label]), kept[label], overlaps[label],
                          len(pools[label]) - positions[label]))
    points = pd.DataFrame(chosen).sort_values(["site_id", "decision_time"]).reset_index(drop=True)
    points["episode_id"] = points.episode_id.astype("Int64")
    return points, pd.DataFrame(audit, columns=["site_id", "reference_label", "eligible_ticks",
        "candidate_points", "selected", "overlap_rejections", "unvisited_after_cap"])


def aggregate_scout_points(db, points, prefix="train"):
    """Fetch the selected windows in site batches and reuse the verified summaries."""
    if prefix not in {"train", "validation"}:
        raise ValueError("Use the training or validation views.")
    features, provenance = [], []
    for site, selected in points.groupby("site_id"):
        db.register("selected_points", selected)
        rows = db.execute(f"""select s.case_id, b.* from {prefix}_bins b join selected_points s
            on b.site_id=s.site_id and b.bin_end>s.decision_time-interval '3 hours'
            and b.bin_end<=s.decision_time where b.site_id=?
            order by s.case_id,b.component,b.observed_at""", [int(site)]).df()
        recent = db.execute(f"""select s.case_id,count(distinct b.component) as recent_components
            from selected_points s join {prefix}_bins b on s.site_id=b.site_id
            and b.bin_end>s.decision_time-interval '24 hours' and b.bin_end<=s.decision_time
            where b.site_id=? group by s.case_id""", [int(site)]).df().set_index("case_id")
        db.unregister("selected_points")
        times = selected.set_index("case_id").decision_time
        for case_id, window in rows.groupby("case_id", sort=False):
            t = times[case_id]
            assert window.observed_at.between(t - LOOKBACK, t, inclusive="right").all()
            result = summarise_window(window)
            count = result["observed_components"]
            result["usable_component_fraction"] = result["usable_components_ch1"] / count
            result["recent_component_fraction"] = count / recent.loc[case_id, "recent_components"]
            result["bin_coverage"] = result["observed_component_bins_ch1"] / (12 * count)
            result["occupied_fraction"] = window.operating_state.eq("Occupied").mean()
            result["setback_fraction"] = window.operating_state.eq("Setback").mean()
            # supplementary statistics make their redundancy with robust summaries inspectable
            grouped = window.groupby("component")[list(CHANNELS)]
            good = grouped.count() >= MIN_BINS
            extras = {"mean": grouped.mean(), "std": grouped.std(ddof=1),
                      "minimum": grouped.min(), "maximum": grouped.max(),
                      "range": grouped.max() - grouped.min()}
            for stat, data in extras.items():
                for ch in CHANNELS:
                    result[f"{ch}_median_of_component_{stat}s"] = data[ch][good[ch]].median()
            features.append({"case_id": case_id, **result})
            provenance.append((case_id, t - LOOKBACK, t, window.observed_at.min(), window.observed_at.max(),
                               int(recent.loc[case_id, "recent_components"])))
        print(f"Summarised site {site} with {len(selected)} scouting windows", flush=True)
    return pd.DataFrame(features), pd.DataFrame(provenance, columns=["case_id", "window_start",
        "window_end", "first_observation", "last_observation", "recent_components"])


def coverage_sensitivity(cases):
    results = []
    for threshold in (0.0, 0.5, 0.75, 0.9, 1.0):
        for label, group in cases.groupby("reference_label"):
            keep = (group.usable_component_fraction >= threshold) & (group.recent_component_fraction >= threshold)
            results.append((threshold, label, len(group), int(keep.sum()), group.loc[keep, "site_id"].nunique()))
    return pd.DataFrame(results, columns=["fraction_required", "reference_label", "scouted", "retained", "sites"])


def correlated_pairs(frame, threshold=0.95):
    corr = frame.corr(method="spearman")
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns)
             for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= threshold]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_r"]).sort_values(
        "spearman_r", key=abs, ascending=False)


class FoldTransform(TransformerMixin, BaseEstimator):
    """Fit optional clipping, medians, and redundancy removal on fitting sites."""

    def __init__(self, clip=False, correlation_limit=0.95):
        self.clip = clip
        self.correlation_limit = correlation_limit

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.lower_ = frame.quantile(0.005).to_numpy() if self.clip else np.full(frame.shape[1], -np.inf)
        self.upper_ = frame.quantile(0.995).to_numpy() if self.clip else np.full(frame.shape[1], np.inf)
        clipped = np.clip(frame.to_numpy(), self.lower_, self.upper_)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        filled = self.imputer_.fit_transform(clipped)
        candidates = np.flatnonzero(np.std(filled, axis=0) > 1e-12)
        corr = pd.DataFrame(filled[:, candidates]).corr(method="spearman").to_numpy()
        retained = []
        for i in range(len(candidates)):
            if not retained or np.all(np.abs(corr[i, retained]) < self.correlation_limit):
                retained.append(i)
        if not retained:
            raise ValueError("The fitting sites have no varying predictors.")
        self.selected_ = candidates[retained]
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X):
        clipped = np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
        return self.imputer_.transform(clipped)[:, self.selected_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.selected_]


def logistic_pipeline(variant):
    clip = variant in {"clipped_linear", "clipped_spline"}
    steps = [("prepare", FoldTransform(clip=clip))]
    if variant == "clipped_spline":
        steps.append(("basis", SplineTransformer(n_knots=4, degree=3, include_bias=False,
                                               knots="quantile", extrapolation="linear")))
    steps.extend([("scale", StandardScaler()),
                  ("logistic", LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs",
                                                  max_iter=2000, class_weight="balanced"))])
    return Pipeline(steps)


def run_diagnostics(cases):
    """Estimate diagnostic performance using five folds containing whole sites."""
    rows, predictions, coefficients = [], [], []
    transforms, memberships, contributions = [], [], []
    split = GroupKFold(n_splits=5)
    y = cases.reference_label.to_numpy()
    site = cases.site_id.to_numpy()
    folds = list(split.split(cases, y, groups=site))
    for fold, (fit, check) in enumerate(folds):
        memberships.extend((fold, int(s), "fit") for s in np.unique(site[fit]))
        memberships.extend((fold, int(s), "check") for s in np.unique(site[check]))
    for variant in ("coverage_only", "linear", "all_numeric", "clipped_linear", "clipped_spline"):
        columns = MODEL_FEATURES
        if variant == "coverage_only":
            columns = list(QUALITY_COLUMNS)
        elif variant == "all_numeric":
            columns = ALL_FEATURES
        for fold, (fit, check) in enumerate(folds):
            assert not set(site[fit]) & set(site[check])
            assert set(y[fit]) == set(LABELS) == set(y[check])
            model = logistic_pipeline(variant)
            started = perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(cases.iloc[fit][columns], y[fit])
                probability = model.predict_proba(cases.iloc[check][columns])
                fit_prediction = model.predict(cases.iloc[fit][columns])
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            classes = model.named_steps["logistic"].classes_
            prediction = classes[probability.argmax(axis=1)]
            per_site = [f1_score(y[check][site[check] == s], prediction[site[check] == s],
                                labels=LABELS, average="macro", zero_division=0) for s in np.unique(site[check])]
            rows.append((variant, fold, len(fit), len(check), len(np.unique(site[check])),
                         model.named_steps["logistic"].coef_.shape[1],
                         f1_score(y[check], prediction, average="macro"),
                         np.mean(per_site), log_loss(y[check], probability, labels=classes),
                         f1_score(y[fit], fit_prediction, average="macro"),
                         int(model.named_steps["logistic"].n_iter_.max()), converged,
                         perf_counter() - started))
            block = pd.DataFrame(probability, columns=[f"score_{c}" for c in classes])
            block.insert(0, "case_id", cases.iloc[check].case_id.to_numpy())
            block["model_id"] = f"scout-v0.2-{variant}-fold{fold}"
            block["fold"] = fold
            block["variant"] = variant
            block["predicted_label"] = prediction
            ordered_scores = np.sort(probability, axis=1)
            block["score_margin"] = ordered_scores[:, -1] - ordered_scores[:, -2]
            predictions.append(block)
            if variant == "linear":
                terms = model.named_steps["prepare"].get_feature_names_out()
                clf = model.named_steps["logistic"]
                prepare = model.named_steps["prepare"]
                scale = model.named_steps["scale"]
                for j, column in enumerate(columns):
                    where = np.flatnonzero(prepare.selected_ == j)
                    k = int(where[0]) if len(where) else None
                    transforms.append((fold, column, k, float(prepare.imputer_.statistics_[j]),
                                       float(scale.mean_[k]) if k is not None else None,
                                       float(scale.scale_[k]) if k is not None else None))
                for class_index, label in enumerate(classes):
                    coefficients.append((fold, label, "(intercept)", float(clf.intercept_[class_index])))
                    coefficients.extend((fold, label, term, float(value))
                                        for term, value in zip(terms, clf.coef_[class_index]))
                transformed = model[:-1].transform(cases.iloc[check][columns])
                rebuilt = softmax(transformed @ clf.coef_.T + clf.intercept_, axis=1)
                assert np.allclose(rebuilt, probability)
                if fold == 0:
                    for class_index, label in enumerate(classes):
                        contributions.append((cases.iloc[check[0]].case_id, fold, label, "(intercept)",
                                              1.0, float(clf.intercept_[class_index]),
                                              float(clf.intercept_[class_index])))
                        contributions.extend((cases.iloc[check[0]].case_id, fold, label, term, float(value),
                                              float(coefficient), float(value * coefficient))
                                             for term, value, coefficient in zip(
                                                 terms, transformed[0], clf.coef_[class_index]))
        print(f"Completed five grouped folds for {variant}", flush=True)
    columns = ["variant", "fold", "fit_cases", "check_cases", "check_sites", "predictors",
               "macro_f1", "site_macro_f1", "log_loss", "fit_macro_f1", "iterations", "converged", "seconds"]
    return {
        "fold_metrics": pd.DataFrame(rows, columns=columns),
        "oof_scores": pd.concat(predictions, ignore_index=True),
        "linear_parameters": pd.DataFrame(coefficients, columns=["fold", "class", "term", "standardised_coefficient"]),
        "linear_preprocessing": pd.DataFrame(transforms, columns=[
            "fold", "feature", "model_column", "imputation_median", "scale_mean", "scale_std"]),
        "fold_sites": pd.DataFrame(memberships, columns=["fold", "site_id", "role"]),
        "example_contributions": pd.DataFrame(contributions, columns=[
            "case_id", "fold", "class", "term", "standardised_value", "coefficient", "contribution"])
    }


def validate_scout(db, cases, predictions):
    """Validate the case contract and the out-of-fold scoring assignments."""
    train = {r[0] for r in db.execute("select site_id from train_inventory").fetchall()}
    assert set(cases.site_id) <= train
    assert cases.case_id.is_unique
    assert cases.episode_id.dropna().is_unique
    assert (cases.decision_time == cases.window_end).all()
    assert (cases.first_observation > cases.window_start).all()
    assert (cases.last_observation <= cases.decision_time).all()
    assert cases.groupby("site_id").decision_time.diff().dropna().ge(LOOKBACK).all()
    assert cases.reference_label.isin(LABELS).all()
    assert cases[list(QUALITY_COLUMNS)].ge(0).all().all()
    assert cases[list(QUALITY_COLUMNS)].le(1).all().all()
    score_columns = [f"score_{c}" for c in LABELS]
    assert predictions.groupby(["variant", "case_id"]).size().eq(1).all()
    assert predictions.groupby("variant").case_id.nunique().eq(len(cases)).all()
    assert np.allclose(predictions[score_columns].sum(axis=1), 1)
    assigned = predictions.merge(cases[["case_id", "site_id"]], on="case_id")
    assert assigned.groupby(["variant", "site_id"]).fold.nunique().eq(1).all()
    return ["Training Sites Only", "Unique Cases and Positive Episodes", "Strict Trailing Windows",
            "Three-Hour Separation Within Sites", "Valid Coverage Fractions",
            "One Out-of-Fold Score Per Case and Variant", "Whole-Site Fold Assignment",
            "Three Scores Sum to One"]
