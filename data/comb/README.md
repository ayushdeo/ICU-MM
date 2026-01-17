# ICU-MM – Processed MIMIC-IV Core Dataset

This folder contains cleaned, analysis-ready tables derived from
**MIMIC-IV v3.1**. These files are intended to be the _canonical structured data inputs_ for the ICU-MM project.

Raw PhysioNet data is **not** included in this repository.

---

## Contents

comb/
├── cohort.csv
├── labs.csv
├── prescriptions.csv
├── respiratory_chartevents.csv
├── respiratory_failurelabels.csv
├── respiratory_procedureevents.csv
└── README.md

---

## cohort.csv (MASTER TABLE)

Defines the ICU cohort and serves as the **join spine** for all other tables.

- One row per ICU stay
- All downstream tables join using `subject_id` and/or `hadm_id`

### Typical uses

- Defining study population
- Temporal alignment
- Patient-level joins

---

## labs.csv

Laboratory measurements during hospital admissions.

- Long format (not pivoted)
- Numeric values only (`valuenum` from MIMIC-IV)
- Filtered to admissions present in `cohort.csv`

### Typical uses

- Feature engineering
- Time-series modeling
- Clinical severity analysis

---

## prescriptions.csv

Medication orders during hospital admissions.

- One row per medication order
- Includes dose, route, and start/stop times
- Filtered to admissions present in `cohort.csv`

### Typical uses

- Treatment pattern analysis
- Medication exposure features
- Confounding adjustment

---

## Notes & Design Decisions

- Tables are intentionally **not merged**
- No outcomes are defined at this stage
- No aggregation or pivoting has been applied
- All timestamps are retained in original resolution
- Imaging (MIMIC-CXR) is added in later stages

---

## Modeling Contract **(Important)**

The following files define model inputs:

- cohort.csv
- labs.csv
- prescriptions.csv

The following files define outcome labels and MUST NOT be used as model inputs:

- respiratory_chartevents.csv
- respiratory_procedureevents.csv
- respiratory_failure_labels.csv

Using label-defining tables as features will cause data leakage.

## Modeling Objectives

Primary outcome:

- Composite clinical deterioration

Secondary outcomes:

- Respiratory failure
- Vasopressor initiation
- ICU mortality

Respiratory failure is the most imaging-aligned outcome and is used
for detailed CXR-focused evaluation and ablation studies.

## Source

Data derived from:

- MIMIC-IV v3.1 (PhysioNet)

All users must comply with the PhysioNet data use agreement.
