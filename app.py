"""
MSME Project Completion & Delinquency Risk Predictor
Streamlit app — Western Visayas, Loan-funded MSME projects

Deployment structure:
    /your-app-folder/
        app.py
        logisticregression.pkcls
        requirements.txt
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

# Hide sidebar entirely
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        .block-container { max-width: 720px; padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants — must match Orange preprocessing order exactly
# ─────────────────────────────────────────────
PROVINCES  = ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros"]
SECTORS    = [
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
OWNERSHIPS = ["Cooperative", "Corporation", "Partnership", "Single"]
SIZES      = ["micro", "small", "medium"]

FEATURE_NAMES = (
    ["Project_Cost", "Year"]
    + [f"Province={p}" for p in PROVINCES]
    + [f"Sector={s}" for s in SECTORS]
    + [f"type_of_ownership={o}" for o in OWNERSHIPS]
    + [f"size_of_enterprise={s}" for s in SIZES]
    + ["Has_Prior_Funding=False", "Has_Prior_Funding=True"]
)

# ─────────────────────────────────────────────
# Model — loaded once at startup
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logisticregression.pkcls")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        orange_model = pickle.load(f)
    return orange_model.skl_model

try:
    skl_model = load_model()
except Exception as e:
    st.error(f"⚠️ Model could not be loaded. Contact your system administrator.\n\n`{e}`")
    st.stop()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def build_feature_vector(year, cost_raw, province, sector, ownership, size, prior_funding):
    X = [cost_raw / 1_000_000, float(year)]
    X += [1 if province == p else 0 for p in PROVINCES]
    X += [1 if sector == s else 0 for s in SECTORS]
    X += [1 if ownership == o else 0 for o in OWNERSHIPS]
    X += [1 if size == s else 0 for s in SIZES]
    X += [1 if prior_funding == "No" else 0]
    X += [1 if prior_funding == "Yes" else 0]
    return np.array(X, dtype=float).reshape(1, -1)


def get_risk_tier(pct: int):
    if pct <= 30: return "Low Risk",    "🟢", "#EAF3DE", "#97C459", "#2B5C0A"
    if pct <= 50: return "Medium Risk", "🟡", "#FAEEDA", "#EF9F27", "#7A4209"
    if pct <= 70: return "High Risk",   "🟠", "#FAECE7", "#F0997B", "#7A2A0A"
    return             "Critical",   "🔴", "#FCEBEB", "#F09595", "#6B1010"


def result_card(label, value, bg, border, text_color):
    st.markdown(
        f"""<div style="background:{bg};border:1.5px solid {border};border-radius:12px;
                        padding:16px 20px;text-align:center;">
                <div style="font-size:12px;font-weight:500;color:#666;
                            text-transform:uppercase;letter-spacing:0.05em;
                            margin-bottom:6px;">{label}</div>
                <div style="font-size:22px;font-weight:600;color:{text_color};">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# UI — header
# ─────────────────────────────────────────────
st.title("MSME Delinquency Risk Assessment")
st.caption("Western Visayas · Loan-funded project screening")
st.divider()

# ─────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────
st.subheader("Applicant details")

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
    size_display = st.selectbox("Enterprise size", [s.capitalize() for s in SIZES])
    size = size_display.lower()
with col7:
    prior_funding = st.selectbox("Prior funding", ["Yes", "No"])

st.divider()

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
if st.button("Assess delinquency risk", use_container_width=True, type="primary"):
    X = build_feature_vector(year, cost_raw, province, sector, ownership, size, prior_funding)

    try:
        pred_class      = skl_model.predict(X)[0]
        proba           = skl_model.predict_proba(X)[0]
        p_completed     = float(proba[0])
        p_not_completed = float(proba[1])
        delinquency_pct = round(p_not_completed * 100)
        completed_pct   = round(p_completed * 100)
        is_completed    = pred_class == 0.0

        tier_name, tier_icon, bg, border, text_color = get_risk_tier(delinquency_pct)

        st.subheader("Risk assessment result")

        # Outcome + tier
        c1, c2 = st.columns(2)
        with c1:
            outcome_icon = "✅" if is_completed else "❌"
            outcome_text = "Completed" if is_completed else "Not Completed"
            result_card("Predicted outcome", f"{outcome_icon} {outcome_text}", bg, border, text_color)
        with c2:
            result_card("Delinquency risk tier", f"{tier_icon} {tier_name}", bg, border, text_color)

        st.markdown("")

        # Delinquency probability meter
        st.markdown(f"**Delinquency probability: {delinquency_pct}%**")
        st.progress(
            delinquency_pct / 100,
            text=f"{tier_icon} {tier_name} — {delinquency_pct}% likelihood of non-completion",
        )

        st.markdown("")

        # Probability breakdown
        p1, p2 = st.columns(2)
        with p1:
            st.metric(label="✅ Probability of completion", value=f"{completed_pct}%")
        with p2:
            st.metric(label="❌ Probability of delinquency", value=f"{delinquency_pct}%")

        # Recommended action
        st.markdown("")
        actions = {
            "Low Risk":    ("✅ Safe to approve.", "#EAF3DE", "#97C459"),
            "Medium Risk": ("⚠️ Approve with close monitoring.", "#FAEEDA", "#EF9F27"),
            "High Risk":   ("🚩 Additional review required before approval.", "#FAECE7", "#F0997B"),
            "Critical":    ("🚨 High default risk — escalate to senior review.", "#FCEBEB", "#F09595"),
        }
        action_text, a_bg, a_border = actions[tier_name]
        st.markdown(
            f"""<div style="background:{a_bg};border:1.5px solid {a_border};
                            border-radius:10px;padding:12px 16px;font-size:14px;
                            font-weight:500;margin-top:4px;">
                    {action_text}
                </div>""",
            unsafe_allow_html=True,
        )

        # Risk tier legend
        st.markdown("")
        st.markdown("""
**Risk tier guide**

| Tier | Delinquency probability | Recommended action |
|------|------------------------|--------------------|
| 🟢 Low Risk     | 0 – 30%   | Safe to approve |
| 🟡 Medium Risk  | 31 – 50%  | Approve with monitoring |
| 🟠 High Risk    | 51 – 70%  | Additional review required |
| 🔴 Critical     | 71 – 100% | High default risk — escalate |
        """)

    except Exception as e:
        st.error(f"Prediction failed. Please check the input values and try again.\n\n`{e}`")
