# ICU-MM

ICU-MM is a multimodal ICU research project built on MIMIC-IV and
MIMIC-CXR. This repository contains data preprocessing pipelines
and processed structured datasets.

## Project Structure

    ICU_MM/
    ├── data/
    │   ├── raw/
    │   │   └── mimiciv/                # PhysioNet data (not tracked)
    │   └── comb/                      # Processed structured data (external)
    │       ├── README.md
    │       └── data_dictionary.md
    ├── scripts/                       # Data build scripts
    │   ├── build_cohort.py
    │   ├── build_labs.py
    │   ├── build_prescriptions.py
    │   ├── build_respiratory_chartevents.py
    │   ├── build_respiratory_procedureevents.py
    │   └── build_respiratory_failure_labels.py
    ├── .gitignore
    ├── LICENSE
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

## Processed Data Access

Due to GitHub file size limits and PhysioNet data usage requirements,
processed MIMIC-IV datasets are not stored directly in this repository.

**Processed ICU-MM datasets (Google Drive)**  
https://drive.google.com/drive/folders/1f2BID6N4WFFaeZdBwUW_KKtSYHURnw0F?usp=sharing

### How to use

1. Download all CSV files from the Google Drive link above
2. Place them under the following directory structure:
