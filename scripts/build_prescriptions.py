import pandas as pd
from pathlib import Path

RAW = Path("data/raw/mimiciv")
OUT = Path("data/comb")
OUT.mkdir(parents=True, exist_ok=True)

# Load cohort
cohort = pd.read_csv(OUT / "cohort.csv")
cohort_hadm_ids = set(cohort["hadm_id"].unique())

print(f"Filtering prescriptions to {len(cohort_hadm_ids)} admissions")

# Load prescriptions
rx = pd.read_csv(RAW / "prescriptions.csv")

# Filter to cohort admissions
rx = rx[rx["hadm_id"].isin(cohort_hadm_ids)]

# Select useful columns
rx = rx[[
    "subject_id",
    "hadm_id",
    "starttime",
    "stoptime",
    "drug",
    "dose_val_rx",
    "dose_unit_rx",
    "route"
]]

# Rename time columns
rx = rx.rename(columns={
    "starttime": "med_starttime",
    "stoptime": "med_stoptime"
})

# Sort for sanity
rx = rx.sort_values(["subject_id", "med_starttime"])

# Save
rx.to_csv(OUT / "prescriptions.csv", index=False)

print(f"prescriptions.csv written with {len(rx)} rows")
