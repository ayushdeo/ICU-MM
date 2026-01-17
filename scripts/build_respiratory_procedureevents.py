import pandas as pd
from pathlib import Path

RAW = Path("data/raw/mimiciv")
OUT = Path("data/comb")
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load cohort ICU stays
# -----------------------------
cohort = pd.read_csv(OUT / "cohort.csv")
cohort_stay_ids = set(cohort["stay_id"].unique())

print(f"Filtering procedures to {len(cohort_stay_ids)} ICU stays")

# -----------------------------
# Load procedureevents
# -----------------------------
proc = pd.read_csv(RAW / "procedureevents.csv")

# Filter to cohort ICU stays
proc = proc[proc["stay_id"].isin(cohort_stay_ids)]

print(f"Remaining rows after ICU filter: {len(proc)}")

# -----------------------------
# Load item dictionary (for labels)
# -----------------------------
d_items = pd.read_csv(RAW / "d_items.csv")
d_items = d_items[["itemid", "label"]]

# Merge to get labels
proc = proc.merge(
    d_items,
    on="itemid",
    how="left"
)

# -----------------------------
# Filter to respiratory procedures
# -----------------------------
RESP_KEYWORDS = [
    "intubation",
    "mechanical ventilation",
    "ventilation",
    "endotracheal"
]

pattern = "|".join(RESP_KEYWORDS)

proc = proc[
    proc["label"].str.contains(
        pattern,
        case=False,
        na=False
    )
]

print(f"Respiratory procedures found: {len(proc)}")

# -----------------------------
# Select final columns
# -----------------------------
proc = proc[[
    "subject_id",
    "stay_id",
    "starttime",
    "endtime",
    "itemid",
    "label"
]]

# Sort for readability
proc = proc.sort_values(
    ["subject_id", "stay_id", "starttime"]
)

# -----------------------------
# Save
# -----------------------------
out_file = OUT / "respiratory_procedureevents.csv"
proc.to_csv(out_file, index=False)

print(f"Saved respiratory_procedureevents.csv with {len(proc)} rows")
