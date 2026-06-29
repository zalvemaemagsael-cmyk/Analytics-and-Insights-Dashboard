"""
MSME Project Completion & Delinquency Risk Predictor
Streamlit app — Western Visayas, Loan-funded MSME projects
Model: Logistic Regression trained in Orange Data Mining (.pkcls)
"""

import streamlit as st
import numpy as np
import pickle
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MSME Delinquency Risk Predictor",
    page_icon="🛡️",
    layout="centered",
)

# ─────────────────────────────────────────────
# Constants — must match Orange preprocessing order exactly
# ─────────────────────────────────────────────
PROVINCES   = ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros"]
SECTORS     = [
    "Agriculture/Marine/Aquaculture",
    "Food Processing",
    "Furniture",
    "Gifts, Decors, Handicrafts",
    "Horticulture and Agriculture",
    "Metals & Engineering",
    "Others",
    "Others (Health Products & Services/Pharma)",
    "Others (Health Products)",
    "Others (ICT)",
    "Others (Laboratory Analysis)",
    "Others (Lime Processing)",
    "Others (Materials Testing and Structural Analysis)",
]
OWNERSHIPS  = ["Cooperative", "Corporation", "Partnership", "Single"]
SIZES       = ["micro", "small", "medium"]

FEATURE_NAMES = (
    ["Project_Cost", "Year"]
    + [f"Province={p}" for p in PROVINCES]
    + [f"Sector={s}" for s in SECTORS]
    + [f"type_of_ownership={o}" for o in OWNERSHIPS]
    + [f"size_of_enterprise={s}" for s in SIZES]
    + ["Has_Prior_Funding=False", "Has_Prior_Funding=True"]
)  # 30 features total


# ─────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        orange_model = pickle.load(f)
    return orange_model


def get_skl_model(orange_model):
    """Extract the underlying sklearn model from an Orange classifier."""
    return orange_model.skl_model


# ─────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────
def build_feature_vector(year, cost_raw, province, sector, ownership, size, prior_funding):
    """
    Converts raw form inputs into the 30-feature one-hot vector
    that matches the Orange preprocessing pipeline.

    Feature order:
      [0]  Project_Cost  (normalized: raw / 1,000,000)
      [1]  Year          (raw integer)
      [2-7]   Province one-hot (6)
      [8-20]  Sector one-hot  (13)
      [21-24] type_of_ownership one-hot (4)
      [25-27] size_of_enterprise one-hot (3)
      [28]  Has_Prior_Funding=False
      [29]  Has_Prior_Funding=True
    """
    cost_norm = cost_raw / 1_000_000

    X = [cost_norm, float(year)]
    X += [1 if province == p else 0 for p in PROVINCES]
    X += [1 if sector == s else 0 for s in SECTORS]
    X += [1 if ownership == o else 0 for o in OWNERSHIPS]
    X += [1 if size == s else 0 for s in SIZES]
    X += [1 if prior_funding == "No" else 0]   # Has_Prior_Funding=False
    X += [1 if prior_funding == "Yes" else 0]  # Has_Prior_Funding=True

    return np.array(X, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────
# Risk tier helper
# ─────────────────────────────────────────────
def get_risk_tier(delinquency_pct: int):
    if delinquency_pct <= 30:
        return "🟢 Low Risk", "success"
    elif delinquency_pct <= 50:
        return "🟡 Medium Risk", "warning"
    elif delinquency_pct <= 70:
        return "🟠 High Risk", "warning"
    else:
        return "🔴 Critical", "error"


# ─────────────────────────────────────────────
# Styling helpers
# ─────────────────────────────────────────────
TIER_COLORS = {
    "🟢 Low Risk":    "#EAF3DE",
    "🟡 Medium Risk": "#FAEEDA",
    "🟠 High Risk":   "#FAECE7",
    "🔴 Critical":    "#FCEBEB",
}
TIER_BORDER = {
    "🟢 Low Risk":    "#97C459",
    "🟡 Medium Risk": "#EF9F27",
    "🟠 High Risk":   "#F0997B",
    "🔴 Critical":    "#F09595",
}


def colored_metric(label, value, bg, border):
    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:14px 18px;text-align:center;margin-bottom:10px;">
            <div style="font-size:12px;font-weight:500;color:#555;margin-bottom:4px;">{label}</div>
            <div style="font-size:24px;font-weight:600;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# App UI
# ─────────────────────────────────────────────
st.title("🛡️ MSME Delinquency Risk Predictor")
st.caption("Western Visayas · Loan-funded project screening · Logistic Regression")
st.divider()

# ── Sidebar: model upload ──────────────────────
with st.sidebar:
    st.header("⚙️ Model")
    uploaded_pkl = st.file_uploader(
        "Upload trained model (.pkcls / .pkl)",
        type=["pkcls", "pkl", "pickle"],
        help="Upload your Orange-trained logistic regression classifier.",
    )

    model = None
    skl_model = None

    if uploaded_pkl is not None:
        tmp_path = f"/tmp/{uploaded_pkl.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_pkl.read())
        try:
            model = load_model(tmp_path)
            skl_model = get_skl_model(model)
            st.success(f"✅ Model loaded: `{uploaded_pkl.name}`")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
    else:
        st.info("Upload your `.pkcls` model file to enable predictions.")

    st.divider()
    st.markdown("**Model details**")
    st.markdown("- Algorithm: Logistic Regression")
    st.markdown("- Training instances: 321")
    st.markdown("- Features: 30 (one-hot encoded)")
    st.markdown("- Target: Completed / Not Completed")
    st.markdown("- Source: Orange Data Mining")

    with st.expander("📋 Feature list"):
        for i, name in enumerate(FEATURE_NAMES):
            st.markdown(f"`{i+1}.` {name}")

# ── Main: applicant form ───────────────────────
st.subheader("📋 Applicant details")
st.caption(
    "Fill in the Pre-PIS form data below. "
    "When connected to your internal system, these fields will auto-populate."
)

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2024, step=1)
with col2:
    cost_raw = st.number_input(
        "Project cost (₱)", min_value=0.0, value=150000.0, step=1000.0, format="%.2f"
    )

col3, col4 = st.columns(2)
with col3:
    province = st.selectbox("Province", PROVINCES)
with col4:
    sector = st.selectbox("Sector", SECTORS)

col5, col6, col7 = st.columns(3)
with col5:
    ownership = st.selectbox("Ownership type", OWNERSHIPS)
with col6:
    size = st.selectbox("Enterprise size", [s.capitalize() for s in SIZES])
    size = size.lower()
with col7:
    prior_funding = st.selectbox("Prior funding", ["Yes", "No"])

st.divider()

# ── Prediction ─────────────────────────────────
predict_btn = st.button("🔍 Assess delinquency risk", use_container_width=True, type="primary")

if predict_btn:
    if skl_model is None:
        st.error("⚠️ Please upload your trained model (.pkcls) in the sidebar first.")
    else:
        X = build_feature_vector(year, cost_raw, province, sector, ownership, size, prior_funding)

        try:
            pred_class = skl_model.predict(X)[0]          # 0.0 = Completed, 1.0 = Not Completed
            proba      = skl_model.predict_proba(X)[0]    # [P(Completed), P(Not Completed)]

            p_completed     = float(proba[0])
            p_not_completed = float(proba[1])
            delinquency_pct = round(p_not_completed * 100)
            completed_pct   = round(p_completed * 100)
            is_completed    = pred_class == 0.0

            tier_label, tier_type = get_risk_tier(delinquency_pct)
            bg     = TIER_COLORS[tier_label]
            border = TIER_BORDER[tier_label]

            # ── Result display ─────────────────
            st.subheader("📊 Risk assessment result")

            r1, r2 = st.columns(2)
            with r1:
                outcome_icon = "✅" if is_completed else "❌"
                outcome_text = "Completed" if is_completed else "Not Completed"
                colored_metric(
                    "Predicted outcome",
                    f"{outcome_icon} {outcome_text}",
                    bg, border,
                )
            with r2:
                colored_metric("Delinquency risk tier", tier_label, bg, border)

            # Delinquency probability meter
            st.markdown("**Delinquency probability**")
            st.progress(
                delinquency_pct / 100,
                text=f"{delinquency_pct}% — {tier_label}",
            )

            # Probability breakdown
            st.markdown("**Probability breakdown**")
            b1, b2 = st.columns(2)
            with b1:
                st.metric("✅ P(Completed)", f"{completed_pct}%")
            with b2:
                st.metric("❌ P(Not Completed)", f"{delinquency_pct}%")

            # Risk tier legend
            with st.expander("📖 Risk tier guide"):
                st.markdown(
                    """
| Tier | Delinquency probability | Recommended action |
|------|------------------------|--------------------|
| 🟢 Low Risk | 0 – 30% | Safe to approve |
| 🟡 Medium Risk | 31 – 50% | Approve with monitoring |
| 🟠 High Risk | 51 – 70% | Additional review required |
| 🔴 Critical | 71 – 100% | High default risk — escalate |
                    """
                )

            # Feature contribution breakdown (top drivers)
            with st.expander("🔬 Top risk drivers (model coefficients)"):
                coef = skl_model.coef_[0]
                contributions = [(FEATURE_NAMES[i], float(coef[i]), float(X[0][i]))
                                 for i in range(len(FEATURE_NAMES))]
                active = [(n, c, v) for n, c, v in contributions if v != 0]
                active.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)

                st.markdown("Features active for this applicant, sorted by influence:")
                for name, coef_val, feat_val in active[:10]:
                    direction = "↑ increases" if coef_val > 0 else "↓ decreases"
                    sign = "🔴" if coef_val > 0 else "🟢"
                    st.markdown(
                        f"{sign} **{name}** — coef `{coef_val:+.4f}` · "
                        f"value `{feat_val:.4f}` · {direction} delinquency risk"
                    )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info(
                "This may be a model compatibility issue. "
                "Ensure the .pkcls file is the logistic regression classifier "
                "trained on the same 30-feature schema."
            )
