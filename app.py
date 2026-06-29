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
import pandas as pd

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MSME Delinquency Risk Predictor",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        .block-container { padding-top: 2rem; }

        /* Assessed table styling */
        .assessed-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .assessed-table th {
            text-align: left; padding: 8px 12px;
            font-size: 11px; font-weight: 600; color: #888;
            text-transform: uppercase; letter-spacing: 0.05em;
            border-bottom: 1.5px solid #e5e7eb;
        }
        .assessed-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #f0f0f0;
            vertical-align: middle;
        }
        .assessed-table tr:last-child td { border-bottom: none; }
        .assessed-table .row-num { color: #888; font-size: 12px; }
        .assessed-table .msme-name { font-weight: 500; color: #1a56db; }
        .badge-delinquent {
            background: #FEF3C7; color: #92400E;
            border: 1px solid #FCD34D;
            padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 500;
        }
        .badge-compliant {
            background: #D1FAE5; color: #065F46;
            border: 1px solid #6EE7B7;
            padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 500;
        }
        .badge-critical {
            background: #FEE2E2; color: #991B1B;
            border: 1px solid #FCA5A5;
            padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 500;
        }
        .badge-medium {
            background: #FAEEDA; color: #7A4209;
            border: 1px solid #EF9F27;
            padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PROVINCES  = ["Aklan", "Antique", "Capiz", "Guimaras", "Iloilo", "Negros"]
SECTORS    = [
    "Agriculture/Marine/Aquaculture", "Food Processing", "Furniture",
    "Gifts, Decors, Handicrafts", "Horticulture and Agriculture",
    "Metals & Engineering", "Others",
    "Others (Health Products & Services/Pharma)", "Others (Health Products)",
    "Others (ICT)", "Others (Laboratory Analysis)", "Others (Lime Processing)",
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
# Model
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logisticregression.pkcls")

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
# Mock database
# ─────────────────────────────────────────────
MOCK_DB = [
    {"id": "APP-2024-001", "name": "Nick's Food Enterprise",       "year": 2024, "cost": 692282,  "province": "Negros",   "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2024-002", "name": "3G Aqua Ventures",             "year": 2024, "cost": 181427,  "province": "Negros",   "sector": "Agriculture/Marine/Aquaculture", "ownership": "Cooperative", "size": "small",  "prior_funding": "Yes"},
    {"id": "APP-2023-003", "name": "4JNG Bakeshop",                "year": 2023, "cost": 304228,  "province": "Iloilo",   "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2023-004", "name": "A&L Wood Craft",               "year": 2023, "cost": 84861,   "province": "Antique",  "sector": "Furniture",                     "ownership": "Single",      "size": "micro",  "prior_funding": "No"},
    {"id": "APP-2024-005", "name": "ADAGE Inc.",                   "year": 2024, "cost": 172971,  "province": "Negros",   "sector": "Others (ICT)",                  "ownership": "Corporation", "size": "small",  "prior_funding": "No"},
    {"id": "APP-2022-006", "name": "ADR Muscovado Products",       "year": 2022, "cost": 68910,   "province": "Antique",  "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2023-007", "name": "AJJJ Bakeshop",                "year": 2023, "cost": 260008,  "province": "Iloilo",   "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2023-008", "name": "ATON Marketing",               "year": 2023, "cost": 473542,  "province": "Antique",  "sector": "Metals & Engineering",           "ownership": "Single",      "size": "small",  "prior_funding": "Yes"},
    {"id": "APP-2022-009", "name": "Able Machine Industries",      "year": 2022, "cost": 228283,  "province": "Negros",   "sector": "Metals & Engineering",           "ownership": "Single",      "size": "small",  "prior_funding": "Yes"},
    {"id": "APP-2024-010", "name": "Ace Food Products",            "year": 2024, "cost": 67095,   "province": "Antique",  "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "No"},
    {"id": "APP-2023-011", "name": "Aesha Frozen Foods",           "year": 2023, "cost": 194117,  "province": "Iloilo",   "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2023-012", "name": "Aklan Piña Clothcraft",        "year": 2023, "cost": 209370,  "province": "Aklan",    "sector": "Gifts, Decors, Handicrafts",     "ownership": "Cooperative", "size": "small",  "prior_funding": "Yes"},
    {"id": "APP-2023-013", "name": "Akoni Homemade Products",      "year": 2023, "cost": 39481,   "province": "Capiz",    "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2024-014", "name": "Antique Development Coop",     "year": 2024, "cost": 148921,  "province": "Antique",  "sector": "Agriculture/Marine/Aquaculture", "ownership": "Cooperative", "size": "small",  "prior_funding": "Yes"},
    {"id": "APP-2023-015", "name": "Ariana Coco Products",         "year": 2023, "cost": 60404,   "province": "Antique",  "sector": "Food Processing",               "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2022-016", "name": "Guimaras Mango Growers Coop", "year": 2022, "cost": 320000,  "province": "Guimaras", "sector": "Horticulture and Agriculture",    "ownership": "Cooperative", "size": "small",  "prior_funding": "No"},
    {"id": "APP-2023-017", "name": "Iloilo Handicraft Center",     "year": 2023, "cost": 95000,   "province": "Iloilo",   "sector": "Gifts, Decors, Handicrafts",     "ownership": "Partnership", "size": "micro",  "prior_funding": "No"},
    {"id": "APP-2024-018", "name": "Capiz Shell Craft Enterprise", "year": 2024, "cost": 157000,  "province": "Capiz",    "sector": "Gifts, Decors, Handicrafts",     "ownership": "Single",      "size": "micro",  "prior_funding": "Yes"},
    {"id": "APP-2022-019", "name": "Western Visayas Metals Inc.",  "year": 2022, "cost": 485000,  "province": "Iloilo",   "sector": "Metals & Engineering",           "ownership": "Corporation", "size": "medium", "prior_funding": "Yes"},
    {"id": "APP-2024-020", "name": "Negros Organic Farms",         "year": 2024, "cost": 212000,  "province": "Negros",   "sector": "Agriculture/Marine/Aquaculture", "ownership": "Cooperative", "size": "small",  "prior_funding": "No"},
]

def search_applicants(query: str):
    if not query or len(query) < 2:
        return []
    q = query.lower()
    return [a for a in MOCK_DB if q in a["name"].lower() or q in a["id"].lower()][:8]

def fetch_applicant(applicant_id: str):
    return next((a for a in MOCK_DB if a["id"] == applicant_id), None)

# ─────────────────────────────────────────────
# Prediction helpers
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
                <div style="font-size:12px;font-weight:500;color:#666;text-transform:uppercase;
                            letter-spacing:0.05em;margin-bottom:6px;">{label}</div>
                <div style="font-size:22px;font-weight:600;color:{text_color};">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

def status_badge(tier_name: str) -> str:
    if tier_name in ("Low Risk",):
        return f'<span class="badge-compliant">Compliant</span>'
    elif tier_name == "Medium Risk":
        return f'<span class="badge-medium">Medium Risk</span>'
    elif tier_name == "High Risk":
        return f'<span class="badge-delinquent">Delinquent</span>'
    else:
        return f'<span class="badge-critical">Critical</span>'

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "selected_applicant" not in st.session_state:
    st.session_state.selected_applicant = None
if "raw_query" not in st.session_state:
    st.session_state.raw_query = ""
if "assessed_log" not in st.session_state:
    st.session_state.assessed_log = []   # list of dicts, one per assessment

# ─────────────────────────────────────────────
# Two-column layout: form left, log right
# ─────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.title("MSME Delinquency Risk Assessment")
    st.caption("Western Visayas · Loan-funded project screening")
    st.divider()

    # ── Search ───────────────────────────────
    st.subheader("Search applicant")
    applicant = st.session_state.selected_applicant

    if applicant:
        col_badge, col_clear = st.columns([5, 1])
        with col_badge:
            st.markdown(
                f"""<div style="background:#F0F4FF;border:1px solid #C7D4F5;border-radius:10px;
                                padding:10px 16px;">
                        <span style="font-size:16px;">🏢</span>
                        <strong style="margin-left:8px;">{applicant['name']}</strong>
                        <span style="font-size:12px;color:#666;margin-left:8px;">
                            {applicant['id']} · Pre-PIS data loaded</span>
                    </div>""",
                unsafe_allow_html=True,
            )
        with col_clear:
            st.markdown("<div style='margin-top:6px'>", unsafe_allow_html=True)
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.selected_applicant = None
                st.session_state.raw_query = ""
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        query_input = st.text_input(
            label="Search applicant",
            value=st.session_state.raw_query,
            placeholder="Type MSME name or application ID…",
            label_visibility="collapsed",
        )
        st.session_state.raw_query = query_input

        if query_input and len(query_input) >= 2:
            results = search_applicants(query_input)
            if results:
                for r in results:
                    col_name, col_btn = st.columns([6, 1])
                    with col_name:
                        st.markdown(
                            f"<div style='padding:4px 4px;'>"
                            f"<span style='font-weight:500;font-size:14px;'>{r['name']}</span>"
                            f"<span style='font-size:12px;color:#888;margin-left:10px;'>"
                            f"{r['id']} · {r['province']} · {r['sector']}</span></div>",
                            unsafe_allow_html=True,
                        )
                    with col_btn:
                        if st.button("Select", key=f"btn_{r['id']}", use_container_width=True):
                            st.session_state.selected_applicant = fetch_applicant(r["id"])
                            st.session_state.raw_query = ""
                            st.rerun()
            else:
                st.caption("No matching applicants found.")

    st.divider()

    # ── Form ─────────────────────────────────
    st.subheader("Applicant details")
    if applicant:
        st.caption("Auto-filled from Pre-PIS record. You may adjust before assessing.")
    else:
        st.caption("Select an applicant above to auto-fill, or enter details manually.")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2030, step=1,
                               value=int(applicant["year"]) if applicant else 2024)
    with col2:
        cost_raw = st.number_input("Project cost (₱)", min_value=0.0, step=1000.0, format="%.2f",
                                   value=float(applicant["cost"]) if applicant else 150000.0)

    col3, col4 = st.columns(2)
    with col3:
        province = st.selectbox("Province", PROVINCES,
            index=PROVINCES.index(applicant["province"]) if applicant and applicant["province"] in PROVINCES else 0)
    with col4:
        sector = st.selectbox("Sector", SECTORS,
            index=SECTORS.index(applicant["sector"]) if applicant and applicant["sector"] in SECTORS else 0)

    col5, col6, col7 = st.columns(3)
    with col5:
        ownership = st.selectbox("Ownership type", OWNERSHIPS,
            index=OWNERSHIPS.index(applicant["ownership"]) if applicant and applicant["ownership"] in OWNERSHIPS else 0)
    with col6:
        size_options = [s.capitalize() for s in SIZES]
        size_val = applicant["size"].capitalize() if applicant else "Micro"
        size_display = st.selectbox("Enterprise size", size_options,
            index=size_options.index(size_val) if size_val in size_options else 0)
        size = size_display.lower()
    with col7:
        prior_options = ["Yes", "No"]
        prior_val = applicant["prior_funding"] if applicant else "Yes"
        prior_funding = st.selectbox("Prior funding", prior_options,
            index=prior_options.index(prior_val) if prior_val in prior_options else 0)

    st.divider()

    # ── Predict ──────────────────────────────
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

            # ── Result display ────────────────
            st.subheader("Risk assessment result")
            if applicant:
                st.caption(f"Assessment for: **{applicant['name']}** ({applicant['id']})")

            r1, r2 = st.columns(2)
            with r1:
                icon = "✅" if is_completed else "❌"
                text = "Completed" if is_completed else "Not Completed"
                result_card("Predicted outcome", f"{icon} {text}", bg, border, text_color)
            with r2:
                result_card("Delinquency risk tier", f"{tier_icon} {tier_name}", bg, border, text_color)

            st.markdown("")
            st.markdown(f"**Delinquency probability: {delinquency_pct}%**")
            st.progress(delinquency_pct / 100,
                        text=f"{tier_icon} {tier_name} — {delinquency_pct}% likelihood of non-completion")

            st.markdown("")
            p1, p2 = st.columns(2)
            with p1:
                st.metric("✅ Probability of completion", f"{completed_pct}%")
            with p2:
                st.metric("❌ Probability of delinquency", f"{delinquency_pct}%")

            st.markdown("")
            actions = {
                "Low Risk":    ("✅ Safe to approve.", "#EAF3DE", "#97C459"),
                "Medium Risk": ("⚠️ Approve with close monitoring.", "#FAEEDA", "#EF9F27"),
                "High Risk":   ("🚩 Additional review required before approval.", "#FAECE7", "#F0997B"),
                "Critical":    ("🚨 High default risk — escalate to senior review.", "#FCEBEB", "#F09595"),
            }
            action_text, a_bg, a_border = actions[tier_name]
            st.markdown(
                f"""<div style="background:{a_bg};border:1.5px solid {a_border};border-radius:10px;
                                padding:12px 16px;font-size:14px;font-weight:500;">
                        {action_text}
                    </div>""",
                unsafe_allow_html=True,
            )

            # ── Append to assessed log ────────
            msme_name = applicant["name"] if applicant else "Manual Entry"
            msme_type = applicant["ownership"] if applicant else ownership

            # Avoid duplicate entries for the same MSME in the same session
            existing_ids = [e["id"] for e in st.session_state.assessed_log]
            entry_id = applicant["id"] if applicant else f"MANUAL-{len(st.session_state.assessed_log)+1}"

            entry = {
                "id": entry_id,
                "name": msme_name,
                "province": province,
                "sector": sector,
                "msme_type": size_display,
                "risk_score": delinquency_pct,
                "tier": tier_name,
                "is_completed": is_completed,
            }

            # Update if already assessed, otherwise append
            if entry_id in existing_ids:
                idx = existing_ids.index(entry_id)
                st.session_state.assessed_log[idx] = entry
            else:
                st.session_state.assessed_log.append(entry)

        except Exception as e:
            st.error(f"Prediction failed. Please check the input values and try again.\n\n`{e}`")

# ─────────────────────────────────────────────
# Right column — assessed MSME log
# ─────────────────────────────────────────────
with right_col:
    st.title("Assessed MSMEs")
    st.caption("All assessments made in this session")
    st.divider()

    log = st.session_state.assessed_log

    if not log:
        st.markdown(
            """<div style="text-align:center;padding:3rem 1rem;color:#aaa;">
                <div style="font-size:40px;margin-bottom:12px;">📋</div>
                <div style="font-size:14px;">No assessments yet.<br>
                Search and assess an MSME on the left to see results here.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        # Summary stats
        total     = len(log)
        compliant = sum(1 for e in log if e["tier"] == "Low Risk")
        at_risk   = total - compliant
        avg_risk  = round(sum(e["risk_score"] for e in log) / total)

        s1, s2, s3 = st.columns(3)
        s1.metric("Total assessed", total)
        s2.metric("At risk", at_risk)
        s3.metric("Avg. delinquency %", f"{avg_risk}%")
        st.markdown("")

        # Clear log button
        if st.button("🗑️ Clear log", use_container_width=False):
            st.session_state.assessed_log = []
            st.rerun()

        st.markdown("")

        # Build HTML table
        rows_html = ""
        for i, e in enumerate(log, start=1):
            badge = status_badge(e["tier"])
            rows_html += f"""
            <tr>
                <td class="row-num">{i}</td>
                <td class="msme-name">{e['name']}</td>
                <td>{e['province']}</td>
                <td>{e['sector']}</td>
                <td>{e['msme_type'].capitalize()}</td>
                <td><strong>{e['risk_score']}%</strong></td>
                <td>{badge}</td>
            </tr>"""

        table_html = f"""
        <table class="assessed-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>MSME</th>
                    <th>Province</th>
                    <th>Sector</th>
                    <th>MSME Type</th>
                    <th>Risk Score %</th>
                    <th>Predicted Status</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>"""

        st.markdown(table_html, unsafe_allow_html=True)
