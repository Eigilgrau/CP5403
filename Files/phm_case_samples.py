"""Build measurement-only examples and independently check the private scores."""

import csv
import random
from statistics import median
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from phm_preprocessing import BRACKET, CHANNELS, CONTROL, DEMO_TIME, LOOKBACK, MIN_BINS, SEED, STEP, TARGETS


def read_window(db, site, time):
    time = pd.Timestamp(time)
    if time != time.floor(STEP):
        raise ValueError("Choose a quarter-hour decision time.")
    rows = db.execute("""select * from component_bins where site_id=?
        and bin_end > ? and bin_end <= ? order by component, observed_at""",
                      [site, time - LOOKBACK, time]).df()
    assert not rows.empty and rows.observed_at.between(time - LOOKBACK, time, inclusive="right").all()
    assert not rows.duplicated(["component", "bin_end"]).any()
    return rows


def summarise_window(rows):
    """Reduce each component first so frequent loggers do not dominate."""
    components = rows.component.nunique()
    features = {"lookback_minutes": 180, "observed_components": components}
    ordered = rows.sort_values(["component", "observed_at"])
    for ch in CHANNELS:
        grouped = ordered.groupby("component")[ch]
        count = grouped.count()
        usable = count >= MIN_BINS
        features[f"usable_components_{ch}"] = int(usable.sum())
        features[f"observed_component_bins_{ch}"] = int(count.sum())
        features[f"possible_component_bins_{ch}"] = int(12 * components)
        summaries = {"median": grouped.median(),
                     "iqr": grouped.quantile(0.75) - grouped.quantile(0.25),
                     "change": grouped.last() - grouped.first()}
        for name, values in summaries.items():
            values = values[usable]
            features[f"{ch}_median_of_component_{name}s"] = values.median()
            features[f"{ch}_lowest_component_{name}"] = values.min()
            features[f"{ch}_highest_component_{name}"] = values.max()
    return features


def build_sample(db):
    """Choose three eligible training windows per class, each from a distinct site."""
    train = [r[0] for r in db.execute(
        "select site_id from site_split where split='train' order by site_id").fetchall()]
    random.Random(SEED).shuffle(train)
    selected = [(1, DEMO_TIME, TARGETS[0])]
    used = {1}
    for label in (*TARGETS, CONTROL):
        needed = 2 if label == TARGETS[0] else 3
        for site in train:
            if site in used:
                continue
            time = db.execute("""select min(decision_time) from private_decisions
                where site_id=? and status='eligible' and reference_label=?""", [site, label]).fetchone()[0]
            if time is None:
                continue
            selected.append((site, pd.Timestamp(time), label))
            used.add(site)
            needed -= 1
            if needed == 0:
                break
        if needed:
            raise ValueError(f"Too few distinct training sites for the {label} sample.")
    queries, scores = [], []
    for site, time, label in selected:
        score = db.execute("""select d.site_id, d.decision_time, d.reference_label, d.episode_id,
            d.usable_ch1, s.split from private_decisions d join site_split s using (site_id)
            where site_id=? and decision_time=? and status='eligible'""", [site, time]).df()
        assert len(score) == 1 and score.iloc[0].reference_label == label and score.iloc[0].split == "train"
        rows = read_window(db, site, time)
        features = summarise_window(rows)
        assert features["usable_components_ch1"] == score.iloc[0].usable_ch1
        case_id = uuid5(NAMESPACE_URL, f"phm2015/case/v1/{site}/{time.isoformat()}").hex
        queries.append({"case_id": case_id, **features})
        scores.append({"case_id": case_id, "site_id": site, "decision_time": time,
                       "window_start": time - LOOKBACK, "window_end": time,
                       "first_observation": rows.observed_at.min(), "last_observation": rows.observed_at.max(),
                       "episode_id": score.iloc[0].episode_id, "reference_label": label,
                       "split": score.iloc[0].split})
    query_frame = pd.DataFrame(queries)
    score_frame = pd.DataFrame(scores)
    score_frame["episode_id"] = score_frame.episode_id.astype("Int64")
    db.execute("create table sample_queries as select * from query_frame")
    db.execute("create table private_sample_scores as select * from score_frame")
    db.execute("""create view private_training_index as
        select p.case_id, p.site_id, p.episode_id, p.reference_label
        from private_sample_scores p join site_split s using (site_id) where s.split='train'""")


def verify_worked_case(db, raw):
    """Reconstruct the guide example from CSV independently of the SQL bin logic."""
    left, right = str(DEMO_TIME - LOOKBACK), str(DEMO_TIME)
    latest = {}
    with (raw / "site1a.csv").open(newline="") as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            component, stamp, ch1 = row[:3]
            if left < stamp <= right:
                # measure bin age directly from the decision time
                age_bin = (DEMO_TIME - pd.Timestamp(stamp)) // STEP
                key = (component, age_bin)
                if key not in latest or (stamp, line) > latest[key][:2]:
                    latest[key] = (stamp, line, float(ch1))
    raw_summary = []
    for component in sorted({key[0] for key in latest}):
        values = [r[2] for (unit, _), r in latest.items() if unit == component]
        raw_summary.append((component, len(values), median(values)))
    check = pd.DataFrame(raw_summary, columns=["component", "raw_bins", "raw_median"])
    rows = read_window(db, 1, DEMO_TIME)
    prepared = rows.groupby("component").ch1.agg(prepared_bins="count", prepared_median="median")
    check = check.join(prepared, on="component", how="outer")
    check["usable"] = check.raw_bins >= MIN_BINS
    assert (check.raw_bins == check.prepared_bins).all()
    assert (check.raw_median == check.prepared_median).all()
    assert check.raw_bins.sum() == 65 and len(check) == 6
    assert check.loc[check.usable, "raw_median"].tolist() == [709, 712, 736, 722.5, 724]

    with (raw / "site1c.csv").open(newline="") as handle:
        active = [(a, b, n) for a, b, n in csv.reader(handle)
                  if n in TARGETS and a <= right and b != r"\N" and right < b]
    assert active == [("2010-06-07 17:59:37", "2010-06-07 18:59:58", TARGETS[0])]
    result = db.execute("""select reference_label, observed_components, usable_components_ch1,
        observed_component_bins_ch1, possible_component_bins_ch1, ch1_median_of_component_medians,
        ch1_lowest_component_median, ch1_highest_component_median from sample_queries
        join private_sample_scores using (case_id) where site_id=1 and decision_time=?""",
                        [DEMO_TIME]).fetchall()
    assert result == [(TARGETS[0], 6, 5, 65, 72, 722.5, 709, 736)]
    return check, pd.DataFrame(active, columns=["raw_start", "raw_end", "recorded_target"])


def validate(db):
    """Check persisted outputs against the local release and the experiment contract."""
    passed = []

    def require(condition, name):
        if not condition:
            raise AssertionError(name)
        passed.append(name)

    def rows(sql, params=()):
        return db.execute(sql, params).fetchall()

    # Source and Clock Checks
    require(rows("""select count(*), count(distinct site_id), countif(component_file='present'),
        countif(zone_file='present'), countif(event_file='present') from site_inventory""")
        == [(70, 70, 66, 58, 69)], "All 70 sites and three stream inventories reconcile")
    require(rows("""select sum(retained_bins)=(select count(*) from component_bins)
        and sum(event_rows)=(select count(*) from private_events)
        and sum(malformed_rows)=(select count(*) from component_rejects)
        and min(superseded_rows)>=0 from site_inventory""") == [(True,)], "Source rows and bin counts reconcile")
    require(rows("""select count(*) from component_bins where
        observed_at > bin_end or observed_at <= bin_end - interval '15 minutes'
        or bin_end != time_bucket(interval '15 minutes', bin_end)""") == [(0,)],
        "Every retained reading lies in its right-closed quarter hour")
    for (site,) in rows("select site_id from site_inventory where retained_bins>0"):
        assert not rows("""select 1 from component_bins where site_id=?
            group by component, bin_end having count(*)>1 limit 1""", [site]), site
    passed.append("Component and bin keys are unique within every site")
    # Episode and Split Checks
    require(not rows("""select 1 from (
        select *, lag(end_at) over (partition by site_id,event_name order by start_at) as prior_end
        from private_episodes) where end_at<=start_at or start_at<prior_end limit 1"""),
        "Merged same-target episodes have positive duration and do not overlap")
    require(rows("""select sum(source_rows)=(select count(*) from private_events
        where quality='valid' and event_name in (?,?)) from private_episodes""", TARGETS) == [(True,)],
        "Every valid target source interval contributes to a merged episode")
    require(not rows("""select 1 from private_decisions d left join private_episodes e using (episode_id)
        where (d.status='eligible') != (d.reference_label is not null)
           or (d.reference_label in (?,?) and (e.episode_id is null or d.site_id!=e.site_id
               or d.reference_label!=e.event_name or d.decision_time<e.start_at or d.decision_time>=e.end_at))
           or (d.status='eligible' and d.usable_ch1<1) limit 1""", TARGETS),
        "Eligible positives match an active episode and every eligible tick meets coverage")
    require(rows("""select count(*), count(distinct site_id),
        countif(split='train'), countif(split='validation'), countif(split='test')
        from site_split""") == [(66, 66, 46, 10, 10)]
        and rows("select split from site_split where site_id=1") == [("train",)],
        "Site assignments are disjoint with Site 1 in training")
    require(rows("""select min(classes) from (select split, count(distinct reference_label) as classes
        from private_decisions join site_split using (site_id) where status='eligible' group by split)""")
        == [(3,)], "All three classes have eligible ticks in each split")
    # Query and Scoring Checks
    require(rows("""select count(*), count(distinct q.case_id), count(distinct s.case_id)
        from sample_queries q full join private_sample_scores s using (case_id)""") == [(9, 9, 9)],
        "Nine measurement queries pair one-to-one with private scores")
    require(not rows("""select 1 from private_sample_scores where decision_time != window_end
        or window_start != decision_time-interval '3 hours'
        or first_observation<=window_start or last_observation>window_end limit 1"""),
        "Every sample window ends at its decision time and contains only available readings")
    require(rows("""select count(distinct site_id), count(episode_id), count(distinct episode_id)
        from private_sample_scores""") == [(9, 6, 6)], "Sample windows are site-separated and positive episodes are unique")
    require(rows("""select count(*), countif(s.split!='train' or s.site_id is null) from private_training_index i
        left join site_split s using (site_id)""") == [(9, 0)], "The sample training index contains only training sites")
    columns = {r[0] for r in rows("describe sample_queries")}
    require(all(c in {"case_id", "lookback_minutes", "observed_components"}
                or c.startswith(CHANNELS) or c.startswith(("usable_components_ch", "observed_component_bins_ch",
                                                         "possible_component_bins_ch")) for c in columns),
            "Model-facing columns contain only the opaque key, coverage, and channel summaries")

    controls = db.execute("select site_id, decision_time from private_sample_scores where reference_label=?", [CONTROL]).df()
    for row in controls.itertuples():
        t = row.decision_time
        counts = rows("""select countif(quality='valid' and start_at>=? and start_at<?),
            countif(quality='valid' and start_at>? and start_at<=?),
            countif(event_name in (?,?,'NOCOMMS') and start_at<=? and (quality!='valid' or end_at>?))
            from private_events where site_id=?""", [t-BRACKET, t, t, t+BRACKET, *TARGETS, t, t, row.site_id])[0]
        assert counts[0] and counts[1] and counts[2] == 0, row
    passed.append("Sample controls are independently bracketed and free of active targets and log outages")
    return passed
