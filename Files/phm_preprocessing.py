"""Prepare the PHM 2015 component and event streams for case construction."""

import random

import numpy as np
import pandas as pd


TARGETS = ("RuntimeExcessive", "SupplyTempTooCold")
CONTROL = "NoRecordedTarget"
CHANNELS = tuple(f"ch{i}" for i in range(1, 8))
SITES = range(1, 71)
SEED = 5403
STEP = pd.Timedelta(minutes=15)
LOOKBACK = 12 * STEP
MIN_BINS = 8
BRACKET = pd.Timedelta(days=7)
DEMO_TIME = pd.Timestamp("2010-06-07 18:30:00")


def csv_source(columns, options=""):
    """Read the headerless schema strictly and retain a deterministic source row."""
    schema = ", ".join(f"'{column}': 'VARCHAR'" for column in columns)
    return (
        f"read_csv(?, columns={{{schema}}}, header=false, auto_detect=false, "
        rf"nullstr=['', '\N'], parallel=false{options}) with ordinality"
    )


def finite_number(column):
    value = f"try_cast({column} as double)"
    return f"case when isfinite({value}) then {value} end"


def grid_ceiling(column):
    # keep readings exactly on a boundary in the bin ending there
    return (f"time_bucket(interval '15 minutes', {column} - interval '1 microsecond') "
            "+ interval '15 minutes'")


def prepare_schema(db):
    channel_types = ", ".join(f"{ch} double" for ch in CHANNELS)
    db.execute(f"""create table component_bins (
        site_id smallint, component varchar, bin_end timestamp, observed_at timestamp,
        source_row bigint, {channel_types}, operating_state varchar)""")
    db.execute("""create table private_events (
        site_id smallint, source_row bigint, event_name varchar, start_at timestamp,
        end_at timestamp, quality varchar)""")
    db.execute("""create table component_rejects (
        site_id smallint, file_line bigint, raw_text varchar, reason varchar)""")
    db.execute("""create table private_decisions (
        site_id smallint, decision_time timestamp, usable_ch1 smallint,
        reference_label varchar, episode_id bigint, status varchar)""")


def ingest_components(db, site, path):
    cols = ("component", "stamp", *CHANNELS, "operating_state")
    options = ", store_rejects=true, rejects_table='rejected', rejects_scan='scanned'"
    values = ", ".join(f"{finite_number(ch)} as {ch}" for ch in CHANNELS)
    invalid = " + ".join(
        f"({ch} is not null and ({finite_number(ch)}) is null)::int" for ch in CHANNELS
    )
    db.execute(f"""create or replace temp table readings as
        select nullif(trim(component), '') as component,
               try_strptime(stamp, '%Y-%m-%d %H:%M:%S') as observed_at,
               ordinality as source_row, {values}, operating_state,
               {invalid} as bad_numeric
        from {csv_source(cols, options)}""", [str(path)])
    db.execute("""insert into component_rejects
        select ?, line, first(csv_line), first(error_message order by column_idx)
        from rejected group by line""", [site])
    db.execute("drop table rejected")
    db.execute("drop table scanned")
    audit = db.execute("""select count(*) as component_rows,
        count(*) filter (where component is null or observed_at is null) as bad_keys,
        coalesce(sum(bad_numeric), 0)::bigint as bad_numeric_cells,
        min(observed_at) filter (where component is not null) as first_reading,
        max(observed_at) filter (where component is not null) as last_reading
        from readings""").df().iloc[0].to_dict()
    audit["malformed_rows"] = db.execute(
        "select count(*) from component_rejects where site_id=?", [site]
    ).fetchone()[0]
    audit["component_rows"] += audit["malformed_rows"]
    ch_list = ", ".join(CHANNELS)
    db.execute(f"""insert into component_bins
        select distinct on (component, bin_end)
               ?, component, {grid_ceiling('observed_at')} as bin_end, observed_at,
               source_row, {ch_list}, operating_state
        from readings where component is not null and observed_at is not null
        order by component, bin_end, observed_at desc, source_row desc""", [site])
    audit["retained_bins"] = db.execute(
        "select count(*) from component_bins where site_id = ?", [site]
    ).fetchone()[0]
    db.execute("drop table readings")
    return audit


def ingest_events(db, site, path):
    source = csv_source(("start_text", "end_text", "name"))
    db.execute(f"""insert into private_events
        select ?, ordinality, nullif(trim(name), '') as event_name,
               try_strptime(start_text, '%Y-%m-%d %H:%M:%S') as start_at,
               try_strptime(end_text, '%Y-%m-%d %H:%M:%S') as end_at,
               case when event_name is null then 'missing_name'
                    when start_at is null then 'invalid_start'
                    when end_text is null then 'missing_end'
                    when end_at is null then 'invalid_end'
                    when end_at <= start_at then 'nonpositive_duration'
                    else 'valid' end as quality
        from {source}""", [site, str(path)])


def inventory_sites(db, raw):
    records = []
    for site in SITES:
        record = {"site_id": site}
        for suffix, stream in zip("abc", ("component", "zone", "event")):
            path = raw / f"site{site}{suffix}.csv"
            size = path.stat().st_size if path.is_file() else None
            record[f"{stream}_bytes"] = size
            record[f"{stream}_file"] = "missing" if size is None else "present" if size else "empty"
        record.update(component_rows=0, malformed_rows=0, bad_keys=0, bad_numeric_cells=0,
                      retained_bins=0, first_reading=None, last_reading=None)
        if record["component_file"] == "present":
            record.update(ingest_components(db, site, raw / f"site{site}a.csv"))
        if record["event_file"] == "present":
            ingest_events(db, site, raw / f"site{site}c.csv")
        records.append(record)
        if site % 10 == 0:
            print(f"Audited {site} of {len(SITES)} sites", flush=True)
    inventory = pd.DataFrame(records)
    db.execute("""create table site_inventory as
        select i.*, component_rows - malformed_rows - bad_keys - retained_bins as superseded_rows,
               coalesce(e.event_rows, 0) as event_rows,
               coalesce(e.valid_events, 0) as valid_events,
               nullif(concat_ws(', ',
                   case when retained_bins = 0 then 'no component bins' end,
                   case when coalesce(valid_events, 0) = 0 then 'no valid log intervals' end,
                   case when unknown_start > 0 then 'unlocatable target start' end
               ), '') as exclusion
        from inventory i left join (
            select site_id, count(*) as event_rows, countif(quality='valid') as valid_events,
                   countif(event_name in (?, ?) and start_at is null) as unknown_start
            from private_events group by site_id
        ) e using (site_id) order by site_id""", list(TARGETS))


def merge_episodes(db):
    """Use the running maximum end to handle nested and overlapping intervals."""
    db.execute(f"""create table private_episodes as
        with ordered as (
            select *, max(end_at) over (
                partition by site_id, event_name order by start_at, end_at, source_row
                rows between unbounded preceding and 1 preceding) as previous_end
            from private_events where quality='valid' and event_name in (?, ?)
        ), islands as (
            select *, sum((previous_end is null or start_at >= previous_end)::int) over (
                partition by site_id, event_name order by start_at, end_at, source_row
                rows unbounded preceding) as island
            from ordered
        ), merged as (
            select site_id, event_name, min(start_at) as start_at, max(end_at) as end_at,
                   count(*) as source_rows from islands group by site_id, event_name, island
        )
        select row_number() over (order by site_id, event_name, start_at) as episode_id, *,
               {grid_ceiling('start_at')} < end_at as hits_clock_grid
        from merged order by site_id, start_at""", list(TARGETS))


def coverage_on_grid(db, site, grid):
    """Count components with at least eight populated ch1 bins at each clock tick."""
    usable = np.zeros(len(grid), dtype=np.int16)
    components = db.execute(
        "select distinct component from component_bins where site_id=?", [site]
    ).fetchall()
    for (component,) in components:
        ends = db.execute("""select bin_end from component_bins
            where site_id=? and component=? and ch1 is not null""", [site, component]).df()
        positions = grid.get_indexer(ends.bin_end)
        present = np.zeros(len(grid), dtype=np.int32)
        present[positions[positions >= 0]] = 1
        counts = present.cumsum()
        # subtract the prefix before the twelve-bin trailing window
        counts[12:] -= counts[:-12]
        usable += counts >= MIN_BINS
    return usable


def control_brackets(grid, starts):
    """Require a distinct valid log start strictly before and after the decision."""
    if len(starts) < 2:
        return np.zeros(len(grid), dtype=bool)
    earlier = starts.searchsorted(grid, side="left") - 1
    later = starts.searchsorted(grid, side="right")
    return ((earlier >= 0) & (later < len(starts))
            & (grid - starts.take(earlier.clip(0)) <= BRACKET)
            & (starts.take(later.clip(max=len(starts) - 1)) - grid <= BRACKET))


def mark_intervals(grid, rows, mask):
    for row in rows.itertuples():
        left = grid.searchsorted(row.start_at)
        right = grid.searchsorted(row.end_at) if row.quality == "valid" else len(grid)
        mask[left:right] = True


def prepare_decisions(db, site, first, last):
    grid = pd.date_range(first.ceil(STEP), last.floor(STEP), freq=STEP)
    usable = coverage_on_grid(db, site, grid)
    events = db.execute("select * from private_events where site_id=?", [site]).df()
    episodes = db.execute("select * from private_episodes where site_id=?", [site]).df()
    active = np.zeros(len(grid), dtype=np.int8)
    episode_ids = np.zeros(len(grid), dtype=np.int64)
    labels = np.full(len(grid), CONTROL, dtype=object)
    for row in episodes.itertuples():
        section = slice(grid.searchsorted(row.start_at), grid.searchsorted(row.end_at))
        active[section] += 1
        episode_ids[section] = row.episode_id
        labels[section] = row.event_name
    valid = events.quality == "valid"
    starts = pd.DatetimeIndex(events.loc[valid, "start_at"].unique()).sort_values()
    bracketed = control_brackets(grid, starts)
    censor = np.zeros(len(grid), dtype=bool)
    outage = np.zeros(len(grid), dtype=bool)
    mark_intervals(grid, events[events.event_name.isin(TARGETS) & ~valid], censor)
    mark_intervals(grid, events[(events.event_name == "NOCOMMS") & events.start_at.notna()], outage)
    # the first matching reason owns the tick in the exclusion report
    status = np.select(
        [censor, active > 1, (active == 0) & outage, (active == 0) & ~bracketed,
         grid - LOOKBACK < first, usable == 0],
        ["unresolved_target", "both_targets_active", "control_log_outage", "control_unbracketed",
         "partial_history", "insufficient_ch1"], default="eligible"
    )
    labels[status != "eligible"] = None
    decisions = pd.DataFrame({"site_id": site, "decision_time": grid, "usable_ch1": usable,
                              "reference_label": labels,
                              "episode_id": pd.Series(episode_ids).where((active == 1) & (status == "eligible")).astype("Int64"),
                              "status": status})
    db.execute("insert into private_decisions select * from decisions")


def freeze_split(db):
    """Allocate whole sites after checking measurement and log alignment."""
    db.execute("""update site_inventory set exclusion='no eligible decisions'
        where exclusion is null and site_id not in
        (select site_id from private_decisions where status='eligible')""")
    eligible = [row[0] for row in db.execute(
        "select site_id from site_inventory where exclusion is null order by site_id"
    ).fetchall()]
    if 1 not in eligible or len(eligible) < 3:
        raise ValueError("Check site eligibility before assigning development and held-out sites.")
    eligible.remove(1)
    random.Random(SEED).shuffle(eligible)
    holdout = max(1, round((len(eligible) + 1) * 0.15))
    groups = {"validation": eligible[:holdout], "test": eligible[holdout:2 * holdout],
              "train": [1, *eligible[2 * holdout:]]}
    manifest = pd.DataFrame([(site, group, SEED) for group, sites in groups.items() for site in sites],
                            columns=["site_id", "split", "seed"])
    db.execute("create table site_split as select * from manifest order by site_id")
    db.execute("""create view private_episode_audit as
        select e.*, coalesce(d.eligible_ticks, 0) as eligible_ticks,
               case when i.exclusion is not null then i.exclusion
                    when not hits_clock_grid then 'misses_grid'
                    when end_at <= i.first_reading or start_at > i.last_reading then 'outside_measurements'
                    when d.eligible_ticks is null then 'no_eligible_tick'
                    else 'available' end as disposition
        from private_episodes e join site_inventory i using (site_id)
        left join (select episode_id, count(*) as eligible_ticks from private_decisions
                   where episode_id is not null group by episode_id) d using (episode_id)""")


