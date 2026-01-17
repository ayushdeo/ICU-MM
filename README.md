# ICU-MM

ICU-MM is a multimodal ICU research project built on MIMIC-IV and
MIMIC-CXR. This repository contains data preprocessing pipelines
and processed structured datasets.

## Project Structure

ICU_MM/
├── data/
│ ├── raw/mimiciv # PhysioNet data (not tracked)
│ └── comb/ # Processed structured data
├── scripts/ # Data build scripts
└── README.md

## Data Build Order

1. Build ICU cohort
2. Filter labs
3. Filter prescriptions
4. (Later) Add vitals and imaging

All data builds are reproducible via scripts in `scripts/`.

**IMP**
This repository does not include large MIMIC-derived datasets.
To reproduce data/comb/\*.csv, place raw MIMIC-IV files under data/raw/ (ignored by git) and run the build scripts in scripts/.
