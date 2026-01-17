import pandas as pd
from pathlib import Path

RAW = Path("data/raw/mimiciv")
OUT = Path("data/comb")
OUT.mkdir(parents=True, exist_ok=True)

# Load tables
patients = pd.read_csv(RAW / "patients.csv")
admissions = pd.read_csv(RAW / "admissions.csv")
icustays = pd.read_csv(RAW / "icustays.csv")

# Merge patients → admissions on subject_id
pat_adm = admissions.merge(
    patients,
    on="subject_id",
    how="inner"
)

# Drop subject_id from icustays to avoid duplication
icustays = icustays.drop(columns=["subject_id"])

# Merge icustays → pat_adm on hadm_id
cohort = icustays.merge(
    pat_adm,
    on="hadm_id",
    how="inner"
)

# Rename columns
cohort = cohort.rename(columns={
    "intime": "icu_intime",
    "outtime": "icu_outtime",
    "gender": "sex"
})

# Select final columns
cohort = cohort[[
    "subject_id",
    "hadm_id",
    "stay_id",
    "icu_intime",
    "icu_outtime",
    "sex",
    "anchor_age"
]]

# Sort
cohort = cohort.sort_values(["subject_id", "icu_intime"])

# Save
cohort.to_csv(OUT / "cohort.csv", index=False)

print(f"cohort.csv written with {len(cohort)} rows")
