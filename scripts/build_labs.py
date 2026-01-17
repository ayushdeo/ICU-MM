import pandas as pd
from pathlib import Path

RAW = Path("data/raw/mimiciv")
OUT = Path("data/comb")
OUT.mkdir(parents=True, exist_ok=True)

# Load cohort
cohort = pd.read_csv(OUT / "cohort.csv")
cohort_hadm_ids = set(cohort["hadm_id"].unique())

print(f"Filtering labs to {len(cohort_hadm_ids)} admissions")

# Load lab dictionary (NO units here in MIMIC-IV)
lab_items = pd.read_csv(RAW / "d_labitems.csv")
lab_items = lab_items[[
    "itemid",
    "label"
]].rename(columns={
    "label": "lab_name"
})

# Output file
out_file = OUT / "labs.csv"
first_chunk = True

chunksize = 1_000_000

for chunk in pd.read_csv(
    RAW / "labevents.csv",
    chunksize=chunksize
):
    # Filter to cohort admissions
    chunk = chunk[chunk["hadm_id"].isin(cohort_hadm_ids)]

    if chunk.empty:
        continue

    # Drop string-valued lab column FIRST
    chunk = chunk.drop(columns=["value"], errors="ignore")

    # Keep numeric values only
    chunk = chunk.dropna(subset=["valuenum"])

    # Merge lab names
    chunk = chunk.merge(
        lab_items,
        on="itemid",
        how="left"
    )

    # Rename columns (NOW safe)
    chunk = chunk.rename(columns={
        "charttime": "lab_time",
        "valuenum": "value",
        "valueuom": "unit"
    })

    # Select final columns
    chunk = chunk[[
        "subject_id",
        "hadm_id",
        "lab_time",
        "lab_name",
        "value",
        "unit"
    ]]


    # Write incrementally
    chunk.to_csv(
        out_file,
        mode="w" if first_chunk else "a",
        index=False,
        header=first_chunk
    )

    first_chunk = False
    print(f"Appended {len(chunk)} rows")

print("labs.csv build complete")
