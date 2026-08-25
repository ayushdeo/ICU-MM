<p align="center">
  <img src="assets/banner.svg" alt="ICU-MM — Multimodal ICU Risk Prediction" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/data-MIMIC--IV%20%2F%20CXR-58a6ff">
  <img src="https://img.shields.io/badge/explainability-SHAP-9b5cff">
</p>

# ICU-MM · Multimodal ICU Risk Prediction

> Predicting **respiratory failure** in ICU patients by fusing three very different signals — structured labs & vitals, free-text radiology reports, and chest X-ray images — into a single, explainable risk score on **MIMIC-IV / MIMIC-CXR**.

The interesting research question isn't just "can we predict it," but **which modalities actually carry the signal** — so every combination is ablated head-to-head on identical splits.

---

## 🧠 Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="Multimodal fusion architecture" width="100%">
</p>

Each modality is embedded independently, PCA-compressed, then **late-fused** into a 219-dimensional vector and classified with logistic regression. Radiology text is embedded with **ClinicalBERT**; chest X-rays with **BioViL**. Predictions are explained per-feature with **SHAP** inside a Streamlit app.

## 📊 Results & modality ablation

Cohort of **552 ICU stays** (338 train / 96 val / 118 test). AUROC on the held-out test set:

| Modalities | Test AUROC |
|---|---|
| Structured + CXR | **0.870** 🥇 |
| Structured only | 0.864 |
| **All three (fusion)** | 0.855 |
| Structured + NLP | 0.853 |
| CXR (BioViL) only | 0.638 |
| NLP + CXR | 0.568 |
| NLP radiology only | 0.536 |

**Takeaways:**
- Structured vitals/labs carry most of the predictive signal (0.864 alone).
- Imaging adds a real, if modest, lift — **Structured + CXR is the strongest combination (0.870)**.
- Radiology *text* alone is near chance (0.536), a useful negative result that stops it from being over-credited.
- Full fusion test **AUROC 0.855 / AUPRC 0.856**.

## 🗂️ Repository structure

```
├── scripts/                # reproducible data-build pipeline (run in order)
│   ├── build_cohort.py                     # define ICU cohort
│   ├── build_labs.py / build_prescriptions.py
│   └── build_respiratory_*.py              # vitals, procedures, failure labels
├── notebooks/
│   ├── ClinicalBERT_Train.ipynb            # radiology-text embeddings
│   ├── GRIDS_FeatureEng_BERT_train.ipynb   # feature engineering
│   └── Fusion_Code_Final.ipynb             # multimodal fusion + ablation
├── models/                 # trained fusion artefacts + fusion_summary.json
└── app/app.py              # Streamlit risk-scoring app with SHAP explanations
```

> Note: MIMIC is credentialed PhysioNet data, so raw and processed datasets are **not** tracked here. The build scripts regenerate everything from raw MIMIC-IV files placed under `data/raw/`.

## ⚙️ Run the app

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## 🧰 Tech stack

`ClinicalBERT` · `BioViL` · `scikit-learn` (PCA + logistic regression) · `SHAP` · `Streamlit` · `pandas` · MIMIC-IV / MIMIC-CXR

---

<sub>Author: **Ayush Deo** · MS CS @ USC · [github.com/ayushdeo](https://github.com/ayushdeo) · Multimodal clinical ML</sub>
