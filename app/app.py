import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ICU-MM | Respiratory Failure Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths — relative to app.py ────────────────────────────────
BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "..", "models")

# ── Load model artefacts (cached) ────────────────────────────
@st.cache_resource
def load_artefacts():
    model  = joblib.load(os.path.join(MODEL_DIR, "fusion_logreg_pca.pkl"))
    pca    = joblib.load(os.path.join(MODEL_DIR, "fusion_pca.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "fusion_scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "fusion_summary.json")) as f:
        summary = json.load(f)
    return model, pca, scaler, summary

model, pca_bundle, scaler, summary = load_artefacts()
pca_nlp      = pca_bundle["pca_nlp"]
pca_cxr      = pca_bundle["pca_cxr"]
nlp_cols     = pca_bundle["nlp_cols"]     # 768 rad_emb_ col names
cxr_cols     = pca_bundle["cxr_cols"]     # 128 emb_ col names
struct_cols  = pca_bundle["structured_cols"]

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='background:#0a2342;padding:18px 24px;border-radius:10px;margin-bottom:8px'>
  <h2 style='color:#e8f4fd;margin:0;font-family:Georgia,serif'>
    🏥 ICU-MM Clinical Decision Support
  </h2>
  <p style='color:#90c4e8;margin:4px 0 0'>
    Multimodal respiratory failure risk assessment · MIMIC-IV/CXR research prototype
  </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "⚠️ **Research use only.** Not validated for clinical decision-making. "
    "Always defer to clinical judgment."
)
st.caption(
    "⚠️ Model performance is most reliable when the ICU data window is set to 12 hours (full observation). "
    "Short windows may underestimate risk due to a training data characteristic — see the Model Performance tab for details."
)

# ── Layout: sidebar (inputs) + main (results) ─────────────────
st.sidebar.header("Patient record")
st.sidebar.caption("Enter values from the patient's current chart.")

# ── Demographics ──────────────────────────────────────────────
st.sidebar.subheader("Demographics")
age             = st.sidebar.slider("Age", 18, 100, 65)
sex_input       = st.sidebar.radio("Sex", ["Male", "Female"])
sex             = 1 if sex_input == "Male" else 0
feature_window  = st.sidebar.slider(
    "Hours of ICU data available", 0.5, 12.0, 6.0, step=0.5,
    help="How many hours since ICU admission. Shorter windows = less data available."
)
is_short_stay   = 1 if feature_window < 12 else 0
obs_window      = min(feature_window, 12.0)

# ── Key labs ──────────────────────────────────────────────────
st.sidebar.subheader("Key lab values")
st.sidebar.caption("Leave at 0 if not available — treated as missing.")

glucose    = st.sidebar.number_input("Glucose (mg/dL)",     0.0, 800.0, 110.0)
sodium     = st.sidebar.number_input("Sodium (mEq/L)",      100.0, 180.0, 138.0)
potassium  = st.sidebar.number_input("Potassium (mEq/L)",   1.0, 10.0, 4.0)
creatinine = st.sidebar.number_input("Creatinine (mg/dL)",  0.0, 20.0, 1.2)
bicarb     = st.sidebar.number_input("Bicarbonate (mEq/L)", 0.0, 50.0, 22.0)
wbc        = st.sidebar.number_input("WBC (K/uL)",          0.0, 100.0, 8.5)
hgb        = st.sidebar.number_input("Hemoglobin (g/dL)",   0.0, 25.0, 12.0)
platelets  = st.sidebar.number_input("Platelets (K/uL)",    0.0, 1000.0, 200.0)

# ── Medications ───────────────────────────────────────────────
st.sidebar.subheader("Active medications")
on_abx       = st.sidebar.checkbox("Antibiotics")
on_sedation  = st.sidebar.checkbox("Sedation / paralytic")
on_opioid    = st.sidebar.checkbox("Opioids")
on_cardiac   = st.sidebar.checkbox("Cardiac medications")
on_anticoag  = st.sidebar.checkbox("Anticoagulation")
on_insulin   = st.sidebar.checkbox("Insulin")
on_diuretic  = st.sidebar.checkbox("Diuretics")
on_steroid   = st.sidebar.checkbox("Corticosteroids")
iv_drip      = st.sidebar.number_input("Number of IV drips running", 0, 20, 0)
total_presc  = st.sidebar.number_input("Total medication orders", 0, 100, 5)
unique_drugs = st.sidebar.number_input("Unique drugs ordered",    0, 50,  3)

# ── CXR section ───────────────────────────────────────────────
st.sidebar.subheader("Chest X-ray")
st.sidebar.caption(
    "In a live system this would load automatically from PACS. "
    "For this prototype, CXR embeddings are simulated as zeros "
    "when no image is provided."
)
cxr_available = st.sidebar.checkbox("CXR available in system", value=True)

# ── Radiology report ──────────────────────────────────────────
st.sidebar.subheader("Radiology report")
rad_report = st.sidebar.text_area(
    "Report text (auto-loaded from RIS in live system)",
    value="Bilateral infiltrates noted. Increased interstitial markings "
          "consistent with early pulmonary edema. No pneumothorax.",
    height=100,
)

# ── Build feature vector ──────────────────────────────────────
def build_feature_vector():
    # Start with NaN everywhere — missing = not measured
    row = {c: np.nan for c in struct_cols}

    # Admin features — always known, never NaN
    admin = {
        "anchor_age":           float(age),
        "sex":                  float(sex),
        "obs_window_hours":     float(obs_window),
        "feature_window_hours": float(feature_window),
        "is_short_stay":        float(is_short_stay),
        "stay_duration_hours":  float(feature_window),
    }
    for k, v in admin.items():
        if k in row:
            row[k] = v

    # Labs — only fill if user entered a non-zero value
    lab_map = {
        "Glucose":           glucose,
        "Sodium":            sodium,
        "Potassium":         potassium,
        "Creatinine":        creatinine,
        "Bicarbonate":       bicarb,
        "White_Blood_Cells": wbc,
        "Hemoglobin":        hgb,
        "Platelet_Count":    platelets,
    }
    for lab, val in lab_map.items():
        if val > 0:
            for stat in ["mean", "min", "max", "first", "last", "early_mean"]:
                k = f"{lab}_{stat}"
                if k in row:
                    row[k] = float(val)
            if f"{lab}_count" in row:
                row[f"{lab}_count"] = 1.0
            if f"{lab}_delta" in row:
                row[f"{lab}_delta"] = 0.0

    # Prescriptions — always known (0 is a real value here)
    presc = {
        "total_presc":       float(total_presc),
        "unique_drugs":      float(unique_drugs),
        "iv_drip_count":     float(iv_drip),
        "iv_drip_ratio":     float(iv_drip) / max(float(total_presc), 1.0),
        "iv_count":          float(iv_drip),
        "has_antibiotic":    float(on_abx),
        "has_sedation":      float(on_sedation),
        "has_opioid":        float(on_opioid),
        "has_cardiac":       float(on_cardiac),
        "has_anticoagulant": float(on_anticoag),
        "has_insulin":       float(on_insulin),
        "has_diuretic":      float(on_diuretic),
        "has_steroid":       float(on_steroid),
    }
    for k, v in presc.items():
        if k in row:
            row[k] = v

    df_row = pd.DataFrame([row])[struct_cols]
    return df_row

# ── Predict ───────────────────────────────────────────────────
predict_btn = st.sidebar.button("🔍 Assess risk", use_container_width=True, type="primary")

# ── Main panel tabs ───────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Patient summary",
    "⚠️ Risk assessment",
    "🔬 Explainability (xAI)",
    "📊 Model performance",
])

with tab1:
    st.subheader("Current patient chart")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**")
        st.write(f"Age: {age} · Sex: {sex_input}")
        st.write(f"ICU data window: {feature_window:.1f}h")
    with col2:
        st.markdown("**Active labs**")
        st.write(f"Glucose: {glucose} · Na: {sodium} · K: {potassium}")
        st.write(f"Creatinine: {creatinine} · HCO₃: {bicarb}")
        st.write(f"WBC: {wbc} · Hgb: {hgb} · Plt: {platelets}")
    with col3:
        st.markdown("**Active medications**")
        active = [n for n, v in [
            ("Antibiotics", on_abx), ("Sedation", on_sedation),
            ("Opioids", on_opioid),  ("Cardiac", on_cardiac),
            ("Anticoag", on_anticoag),("Insulin", on_insulin),
            ("Diuretics", on_diuretic),("Steroids", on_steroid),
        ] if v]
        st.write(", ".join(active) if active else "None flagged")
        st.write(f"Total orders: {total_presc} · IV drips: {iv_drip}")

    st.divider()
    st.markdown("**Radiology report (from RIS)**")
    st.info(rad_report)

with tab2:
    if not predict_btn:
        st.info("Fill in the patient chart on the left and click **Assess risk** to run the model.")
    else:
        X_struct = build_feature_vector()

        # Fill NaN with training medians (stored in scaler)
        # The scaler mean_ approximates the center — use it as fill
        all_feature_names = list(scaler.feature_names_in_) \
            if hasattr(scaler, 'feature_names_in_') \
            else (struct_cols + nlp_cols + cxr_cols)

        X_full = np.zeros((1, len(all_feature_names)))
        scaler_means = scaler.mean_  # training means ≈ good fill for missing

        # Fill structured positions with training mean (for NaN cols)
        # and actual values (for entered cols)
        for i, col in enumerate(struct_cols):
            if col in all_feature_names:
                pos = all_feature_names.index(col)
                val = X_struct[col].values[0]
                if np.isnan(val):
                    X_full[0, pos] = scaler_means[pos]  # use training mean
                else:
                    X_full[0, pos] = val

        X_full_scaled = scaler.transform(X_full)

        # Extract the three modality blocks from the scaled vector
        str_idx_in_full = [all_feature_names.index(c) for c in struct_cols
                        if c in all_feature_names]
        nlp_idx_in_full = [all_feature_names.index(c) for c in nlp_cols
                        if c in all_feature_names]
        cxr_idx_in_full = [all_feature_names.index(c) for c in cxr_cols
                        if c in all_feature_names]

        X_scaled_struct = X_full_scaled[:, str_idx_in_full]

        # NLP and CXR embeddings — zeros (no live embedding extraction in UI)
        nlp_vec = np.zeros((1, len(nlp_cols)))
        cxr_vec = np.zeros((1, len(cxr_cols)))

        nlp_pca_vec = pca_nlp.transform(nlp_vec)
        cxr_pca_vec = pca_cxr.transform(cxr_vec)

        # Fused vector: 179 structured + 20 NLP PCA + 20 CXR PCA
        X_fused = np.hstack([X_scaled_struct, nlp_pca_vec, cxr_pca_vec])

        fw_idx = [i for i, c in enumerate(struct_cols) if 'feature_window' in c]
        
        prob  = float(model.predict_proba(X_fused)[0, 1])
        label = "HIGH RISK" if prob > 0.7 else "MODERATE RISK" if prob > 0.4 else "LOW RISK"
        color = "#c0392b" if prob > 0.7 else "#e67e22" if prob > 0.4 else "#27ae60"

        # Risk banner — uses prob directly
        st.markdown(f"""
        <div style='background:{color}18;border:2px solid {color};
                    border-radius:12px;padding:24px;text-align:center;margin-bottom:16px'>
        <div style='font-size:3rem;font-weight:700;color:{color}'>{prob:.1%}</div>
        <div style='font-size:1.3rem;color:{color};font-weight:600'>{label}</div>
        <div style='color:#555;margin-top:6px'>
            Probability of respiratory failure
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Clinical guidance
        st.subheader("Clinical guidance")
        if prob > 0.7:
            st.error(
                "**Immediate review recommended.** This patient has a high predicted "
                "probability of respiratory failure. Consider escalating monitoring, "
                "reviewing ventilation parameters, and notifying the attending physician."
            )
        elif prob > 0.4:
            st.warning(
                "**Increased vigilance advised.** Moderate risk of deterioration. "
                "Increase monitoring frequency and reassess in 1–2 hours."
            )
        else:
            st.success(
                "**Low risk at this time.** Continue routine monitoring per protocol. "
                "Reassess if clinical status changes."
            )

        # Risk gauge bar
        fig, ax = plt.subplots(figsize=(7, 1.2))
        ax.barh(0, prob,       color=color,   height=0.4)
        ax.barh(0, 1 - prob, left=prob, color="#ecf0f1", height=0.4)
        ax.axvline(0.4, color="#e67e22", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.axvline(0.7, color="#c0392b", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_xlim(0, 1); ax.set_yticks([])
        ax.set_xlabel("Predicted probability")
        ax.set_title("Risk gauge")
        ax.text(0.2,  0.55, "Low",      ha="center", fontsize=9, color="#27ae60")
        ax.text(0.55, 0.55, "Moderate", ha="center", fontsize=9, color="#e67e22")
        ax.text(0.85, 0.55, "High",     ha="center", fontsize=9, color="#c0392b")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Store for xAI tab
        st.session_state["X_fused"]  = X_fused
        st.session_state["X_struct"] = X_scaled_struct
        st.session_state["prob"]     = prob
        st.session_state["struct_cols"] = struct_cols

with tab3:
    st.subheader("Explainability — why this prediction?")

    if "X_fused" not in st.session_state:
        st.info("Run the risk assessment first (Tab 2) to see explanations.")
    else:
        X_fused     = st.session_state["X_fused"]
        prob        = st.session_state["prob"]
        s_cols      = st.session_state["struct_cols"]

        st.caption(
            "SHAP values show how much each feature pushed the prediction "
            "toward (red) or away from (blue) respiratory failure."
        )

        with st.spinner("Computing SHAP values..."):
            # Use linear explainer — fast for LogReg
            # Background = zero vector (mean after scaling)
            background = np.zeros((1, X_fused.shape[1]))
            explainer  = shap.LinearExplainer(model, background,
                                               feature_perturbation="interventional")
            shap_vals  = explainer.shap_values(X_fused)[0]  # shape: (219,)

        # Build feature name list matching the 219-dim fused vector
        fused_names = (
            list(s_cols) +
            [f"NLP_PC{i+1}" for i in range(pca_nlp.n_components_)] +
            [f"CXR_PC{i+1}" for i in range(pca_cxr.n_components_)]
        )

        # Top 20 by |SHAP|
        top_idx   = np.argsort(np.abs(shap_vals))[::-1][:20]
        top_names = [fused_names[i] for i in top_idx]
        top_vals  = shap_vals[top_idx]

        # Clean up lab names for display
        def clean_name(n):
            return (n.replace("_mean","").replace("_"," ")
                     .replace("White Blood Cells","WBC")
                     .replace("Platelet Count","Platelets")
                     .replace("Urea Nitrogen","BUN"))

        display_names = [clean_name(n) for n in top_names]
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in top_vals]

        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(range(len(top_vals)), top_vals[::-1],
                       color=colors[::-1], edgecolor="white", height=0.7)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(display_names[::-1], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title("Top 20 features driving this prediction")
        red_patch  = mpatches.Patch(color="#c0392b", label="Increases RF risk")
        blue_patch = mpatches.Patch(color="#2980b9", label="Decreases RF risk")
        ax.legend(handles=[red_patch, blue_patch], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Narrative explanation
        top_risk    = [display_names[i] for i, v in enumerate(top_vals) if v > 0][:3]
        top_protect = [display_names[i] for i, v in enumerate(top_vals) if v < 0][:3]

        st.markdown("**Narrative summary**")
        if top_risk:
            st.markdown(
                f"The prediction is primarily driven upward by: "
                f"**{', '.join(top_risk)}**."
            )
        if top_protect:
            st.markdown(
                f"The following features are reducing the predicted risk: "
                f"**{', '.join(top_protect)}**."
            )

        st.caption(
            "NLP_PC and CXR_PC refer to the principal components of the radiology "
            "report embeddings and chest X-ray embeddings respectively. "
            "Their contribution reflects patterns learned from the MIMIC-CXR corpus."
        )

with tab4:
    st.subheader("Model performance on held-out test set")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test AUROC",  f"{summary.get('test_auroc', 0.8549):.4f}")
    col2.metric("Test AUPRC",  f"{summary.get('test_auprc', 0.8560):.4f}")
    col3.metric("Test samples", str(summary.get("n_test", 118)))
    col4.metric("Total patients", str(summary.get("n_samples", 552)))

    # ROC curve image
    roc_path = os.path.join(BASE, "assets", "roc_curve_fusion.png")
    if os.path.exists(roc_path):
        st.image(roc_path, caption="ROC curve — fusion model (held-out test set)",
                 use_column_width=True)

    # Ablation table
    st.subheader("Ablation — contribution by modality")
    ablation = summary.get("ablation", {})
    if ablation:
        rows = []
        for exp, vals in ablation.items():
            rows.append({
                "Experiment": exp,
                "Val AUROC":  vals["val_auroc"],
                "Test AUROC": vals["test_auroc"],
            })
        abl_df = pd.DataFrame(rows).sort_values("Test AUROC", ascending=False)
        st.dataframe(abl_df, use_container_width=True, hide_index=True)

    st.caption(
        "Fusion model: Logistic Regression on 179 structured features + "
        "20 PCA dims from radiology report embeddings + "
        "20 PCA dims from BioViL-T CXR embeddings. "
        "Trained on 552 patients with all three modalities from MIMIC-IV/MIMIC-CXR."
    )