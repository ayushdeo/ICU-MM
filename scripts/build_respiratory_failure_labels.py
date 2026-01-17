import pandas as pd
from pathlib import Path

OUT = Path("data/comb")

# -----------------------------
# Load core tables
# -----------------------------
cohort = pd.read_csv(OUT / "cohort.csv", parse_dates=["icu_intime"])
resp_chart = pd.read_csv(
    OUT / "respiratory_chartevents.csv",
    parse_dates=["charttime"]
)
resp_proc = pd.read_csv(
    OUT / "respiratory_procedureevents.csv",
    parse_dates=["starttime"]
)

# -----------------------------
# 48-hour window
# -----------------------------
WINDOW_HOURS = 48

cohort["window_end"] = cohort["icu_intime"] + pd.Timedelta(hours=WINDOW_HOURS)

# -----------------------------
# Criterion A: invasive ventilation
# -----------------------------
proc = resp_proc.merge(
    cohort[["stay_id", "icu_intime", "window_end", "subject_id"]],
    on="stay_id",
    how="inner"
)

proc = proc[
    (proc["starttime"] >= proc["icu_intime"]) &
    (proc["starttime"] <= proc["window_end"])
]

proc_events = (
    proc.groupby("stay_id")["starttime"]
    .min()
    .reset_index()
    .rename(columns={"starttime": "rf_time"})
)

proc_events["criterion"] = "invasive_vent"

# -----------------------------
# Criterion B & C: chart events
# -----------------------------
chart = resp_chart.merge(
    cohort[["stay_id", "icu_intime", "window_end", "subject_id"]],
    on="stay_id",
    how="inner"
)

chart = chart[
    (chart["charttime"] >= chart["icu_intime"]) &
    (chart["charttime"] <= chart["window_end"])
]

# Criterion B: oxygen device escalation
OXYGEN_KEYWORDS = [
    "high flow",
    "hfnc",
    "bipap",
    "cpap"
]

oxygen_events = chart[
    chart["value"].astype(str).str.lower()
    .str.contains("|".join(OXYGEN_KEYWORDS), na=False)
]

oxygen_events = (
    oxygen_events.groupby("stay_id")["charttime"]
    .min()
    .reset_index()
    .rename(columns={"charttime": "rf_time"})
)

oxygen_events["criterion"] = "oxygen_escalation"

# Criterion C: FiO2 >= 0.6
fio2_events = chart[
    (chart["label"].str.lower() == "fio2") &
    (pd.to_numeric(chart["value"], errors="coerce") >= 0.6)
]

fio2_events = (
    fio2_events.groupby("stay_id")["charttime"]
    .min()
    .reset_index()
    .rename(columns={"charttime": "rf_time"})
)

fio2_events["criterion"] = "high_fio2"

# -----------------------------
# Combine all criteria
# -----------------------------
events = pd.concat(
    [proc_events, oxygen_events, fio2_events],
    ignore_index=True
)

# Earliest respiratory failure per stay
rf = (
    events.groupby("stay_id")["rf_time"]
    .min()
    .reset_index()
)

rf["respiratory_failure"] = 1

# -----------------------------
# Merge back to cohort
# -----------------------------
labels = cohort.merge(
    rf,
    on="stay_id",
    how="left"
)

labels["respiratory_failure"] = labels["respiratory_failure"].fillna(0).astype(int)

labels = labels[[
    "subject_id",
    "stay_id",
    "respiratory_failure",
    "rf_time"
]]

# -----------------------------
# Save
# -----------------------------
out_file = OUT / "respiratory_failure_labels.csv"
labels.to_csv(out_file, index=False)

print(
    f"Respiratory failure labels saved "
    f"({labels['respiratory_failure'].mean():.2%} positive)"
)
