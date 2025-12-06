# # dashboard.py
# """
# Streamlit Dashboard for Payer Data Quality & Anomaly Detection
# Save as `dashboard.py` and run:
#     streamlit run dashboard.py
# This dashboard expects `final_anomaly_severity.csv` to exist in the same folder.
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import os

# st.set_page_config(page_title="Payer Data Quality & Anomaly Dashboard",
#                    layout="wide",
#                    initial_sidebar_state="expanded")

# DATA_FILE = "final_anomaly_severity.csv"

# # ---------------------- Load data (cached) ----------------------
# @st.cache_data
# def load_data(path=DATA_FILE):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"{path} not found. Run pipeline stages 1-5 first.")
#     df = pd.read_csv(path)
#     return df

# try:
#     df = load_data()
# except Exception as e:
#     st.title("Payer Data Quality & Anomaly Dashboard")
#     st.error(str(e))
#     st.stop()

# # Normalize expected columns existence (safety)
# def ensure_col(df, col, default=0.0):
#     if col not in df.columns:
#         df[col] = default
#     return df

# # ensure key scoring columns exist
# df = ensure_col(df, "final_severity_score", 0.0)
# df = ensure_col(df, "final_severity_label", "CLEAN")
# df = ensure_col(df, "combined_score", 0.0)
# df = ensure_col(df, "dq_score", 0.0)
# df = ensure_col(df, "final_anomaly_flag", 0)
# # claim amount column fallback
# if "ClaimAmount" not in df.columns and "ClaimAmount_NUM" in df.columns:
#     df["ClaimAmount"] = df["ClaimAmount_NUM"]
# elif "ClaimAmount" not in df.columns:
#     df["ClaimAmount"] = np.nan

# # ---------------------- Page Header ----------------------
# st.title("Payer Data Quality & Anomaly Detection Dashboard")
# st.markdown("A consolidated view of data-quality issues and unsupervised anomaly detection results. "
#             "Use the sidebar to filter and drill down into flagged claims.")

# # ---------------------- Summary KPIs ----------------------
# st.header("Overall Summary")
# col1, col2, col3, col4, col5 = st.columns(5)
# col1.metric("Total Claims", f"{len(df):,}")
# col2.metric("CRITICAL", int((df["final_severity_label"] == "CRITICAL").sum()))
# col3.metric("HIGH", int((df["final_severity_label"] == "HIGH").sum()))
# col4.metric("MEDIUM", int((df["final_severity_label"] == "MEDIUM").sum()))
# col5.metric("LOW", int((df["final_severity_label"] == "LOW").sum()))

# # ---------------------- Sidebar Filters ----------------------
# st.sidebar.header("Filters")
# severity_options = sorted(df["final_severity_label"].unique(), reverse=True)
# severity_filter = st.sidebar.multiselect(
#     "Select severity levels:",
#     severity_options,
#     default=severity_options
# )

# provider_filter = st.sidebar.text_input("ProviderID contains (optional)", "")
# patient_filter  = st.sidebar.text_input("PatientID contains (optional)", "")
# claimtype_filter = st.sidebar.multiselect(
#     "ClaimType (optional)",
#     options=sorted(df["ClaimType"].dropna().unique()) if "ClaimType" in df.columns else [],
#     default=[]
# )
# state_filter = st.sidebar.multiselect(
#     "State (optional)",
#     options=sorted(df["State"].dropna().unique()) if "State" in df.columns else [],
#     default=[]
# )

# # ---------------------- Apply Filters ----------------------
# df_filtered = df[df["final_severity_label"].isin(severity_filter)]

# if provider_filter:
#     if "ProviderID" in df_filtered.columns:
#         df_filtered = df_filtered[df_filtered["ProviderID"].astype(str).str.contains(provider_filter, na=False)]
# if patient_filter:
#     if "PatientID" in df_filtered.columns:
#         df_filtered = df_filtered[df_filtered["PatientID"].astype(str).str.contains(patient_filter, na=False)]
# if claimtype_filter:
#     if "ClaimType" in df_filtered.columns and len(claimtype_filter) > 0:
#         df_filtered = df_filtered[df_filtered["ClaimType"].isin(claimtype_filter)]
# if state_filter:
#     if "State" in df_filtered.columns and len(state_filter) > 0:
#         df_filtered = df_filtered[df_filtered["State"].isin(state_filter)]

# # ---------------------- DQ Overview ----------------------
# st.header(" Data Quality Issue Overview")

# # detect DQ flag columns heuristically
# dq_flag_cols = [c for c in df.columns if (
#     c.startswith("bad_") or 
#     c.startswith("flag_") or
#     ("dup" in c.lower()) or
#     c in ["missing_key","dq_score","dup_full_row","dup_claimid_flag","amount_suspect_flag"]
# )]
# dq_flag_cols = [c for c in dq_flag_cols if c in df.columns]

# if dq_flag_cols:
#     dq_summary = df[dq_flag_cols].sum().sort_values(ascending=False).reset_index()
#     dq_summary.columns = ["DQ_Issue", "Count"]
#     fig_dq = px.bar(dq_summary, x="DQ_Issue", y="Count", title="Data Quality Issues (counts)", text_auto=True)
#     st.plotly_chart(fig_dq, use_container_width=True)
#     st.dataframe(dq_summary, use_container_width=True)
# else:
#     st.info("No DQ flag columns detected in the data. Expected columns like 'bad_*' or 'flag_*' or 'dq_score'.")

# # ---------------------- ML Score Distributions ----------------------
# st.header(" ML Score Distributions")
# score_cols = [c for c in ["if_score_norm","lof_score_norm","hbos_score_norm","combined_score"] if c in df.columns]
# # fallback to raw normalized versions if not present
# if not score_cols:
#     score_cols = [c for c in ["if_score_norm","lof_score_norm","combined_score"] if c in df.columns]

# if score_cols:
#     for col in score_cols:
#         fig = px.histogram(df, x=col, nbins=60, title=f"Distribution of {col}")
#         st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("No ML score columns found. Ensure Stage 4 output contains model score columns.")

# # ---------------------- ML vs DQ Scatterplot ----------------------
# st.header(" ML Score vs. DQ Score")
# if "combined_score" in df.columns and "dq_score" in df.columns:
#     fig_scatter = px.scatter(
#         df, x="combined_score", y="dq_score",
#         color="final_severity_label",
#         hover_data=["ClaimID","ProviderID","PatientID","ClaimAmount"],
#         title="ML Anomaly Score vs DQ Score"
#     )
#     fig_scatter.update_traces(marker=dict(size=7, opacity=0.6))
#     st.plotly_chart(fig_scatter, use_container_width=True)
# else:
#     st.info("combined_score or dq_score missing for ML vs DQ plot.")

# # ---------------------- Severity Distribution ----------------------
# st.header("Final Severity Distribution")
# sev_counts = df["final_severity_label"].value_counts().reset_index()
# sev_counts.columns = ["Severity", "Count"]
# fig_sev = px.bar(sev_counts, x="Severity", y="Count", color="Severity", title="Final Severity Label Counts", text_auto=True)
# st.plotly_chart(fig_sev, use_container_width=True)

# # histogram of final_severity_score
# if "final_severity_score" in df.columns:
#     fig_score = px.histogram(df, x="final_severity_score", nbins=50, title="Final Severity Score Distribution")
#     st.plotly_chart(fig_score, use_container_width=True)

# # ---------------------- Provider Hotspots ----------------------
# st.header("Provider Hotspots (CRITICAL + HIGH)")
# hot = df[df["final_severity_label"].isin(["CRITICAL","HIGH"])]
# if "ProviderID" in hot.columns:
#     prov_hotspots = hot.groupby("ProviderID").size().reset_index(name="AnomalyCount").sort_values("AnomalyCount", ascending=False)
#     fig_hot = px.bar(prov_hotspots.head(20), x="ProviderID", y="AnomalyCount", title="Top Providers with High Severity Anomalies")
#     st.plotly_chart(fig_hot, use_container_width=True)
# else:
#     st.info("ProviderID not available in data for hotspot analysis.")

# # ---------------------- Categorical Breakdowns ----------------------
# st.header(" Anomaly Breakdown by Categorical Features")
# cat_cols = ["ClaimType","State","PatientGender","ProviderSpecialty"]
# for col in cat_cols:
#     if col in df.columns:
#         temp = df.groupby([col, "final_severity_label"]).size().reset_index(name="Count")
#         fig = px.bar(temp, x=col, y="Count", color="final_severity_label", title=f"Anomalies by {col}")
#         st.plotly_chart(fig, use_container_width=True)

# # ---------------------- Claim Amount Distribution ----------------------
# st.header(" Claim Amount Distribution (Anomalies vs Normal)")
# if "ClaimAmount" in df.columns:
#     fig_amt = px.histogram(df, x="ClaimAmount", color="final_severity_label", nbins=80, title="Claim Amount Distribution by Severity")
#     st.plotly_chart(fig_amt, use_container_width=True)
# else:
#     st.info("ClaimAmount column missing.")

# # ---------------------- Top Anomalies Table ----------------------
# st.header(" Top Anomalies")
# top_k = st.slider("Number of top anomalies to display", 5, 100, 20)
# top = df.sort_values("final_severity_score", ascending=False).head(top_k)
# st.dataframe(top, use_container_width=True)

# # ---------------------- Drilldown by ClaimID ----------------------
# st.header("Claim Drilldown")
# claim_search = st.text_input("Enter ClaimID to inspect (exact match):", "")
# if claim_search:
#     found = df[df["ClaimID"].astype(str) == str(claim_search)]
#     if found.empty:
#         st.warning("No claim found with that ClaimID.")
#     else:
#         st.success("Claim found:")
#         st.dataframe(found.transpose(), use_container_width=True)

# # # ---------------------- Download Filtered Results ----------------------
# # st.header("⬇️ Export")
# # st.markdown("Download the currently filtered dataset (applies sidebar filters).")
# # csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
# # st.download_button(label="Download filtered anomalies (CSV)", data=csv_bytes, file_name="filtered_anomalies.csv", mime="text/csv")

# # ---------------------- Helpful Notes & Report Links ----------------------
# # st.markdown("---")
# # st.markdown("#### Notes")
# # st.markdown(
# #     "- `dq_score` is the normalized fraction of rule-based DQ flags per row (0–1).  \n"
# #     "- `combined_score` is the ML ensemble anomaly score (0–1).  \n"
# #     "- `final_severity_score` blends ML + DQ (pipeline Stage 5).  \n"
# #     "- Use the filters on the left to focus the investigation (ProviderID, ClaimType, State)."
# # )

# # st.markdown("#### Next steps / Exports")
# # st.markdown(
# #     "- Export `filtered_anomalies.csv` for handoff or remediation.  \n"
# #     "- Use the top anomalies list for manual audit and root-cause analysis.  \n"
# #     "- Consider periodic runs and an automated alert for CRITICAL anomalies."
# # )

# # st.markdown("---")
# # st.markdown("Built for the Data Quality Anomaly Detector project — streamlined for investigation and reporting.")


# dashboard.py
"""
Streamlit Dashboard for Payer Data Quality and Anomaly Detection
Save this file as `dashboard.py` and run using:
    streamlit run dashboard.py

This dashboard expects `final_anomaly_severity.csv` to be present 
in the same directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="Payer Data Quality and Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "final_anomaly_severity.csv"

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please run all processing stages first.")
    df = pd.read_csv(path)
    return df

try:
    df = load_data()
except Exception as error:
    st.title("Payer Data Quality and Anomaly Detection Dashboard")
    st.error(str(error))
    st.stop()

# Ensure required columns exist
def ensure_col(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    return df

df = ensure_col(df, "final_severity_score", 0.0)
df = ensure_col(df, "final_severity_label", "CLEAN")
df = ensure_col(df, "combined_score", 0.0)
df = ensure_col(df, "dq_score", 0.0)
df = ensure_col(df, "final_anomaly_flag", 0)

if "ClaimAmount" not in df.columns and "ClaimAmount_NUM" in df.columns:
    df["ClaimAmount"] = df["ClaimAmount_NUM"]
elif "ClaimAmount" not in df.columns:
    df["ClaimAmount"] = np.nan

# ---------------------- Page Header ----------------------
st.title("Payer Data Quality and Anomaly Detection Dashboard")
st.markdown(
    "This dashboard provides a complete overview of detected anomalies, data quality issues, "
    "claim severity distribution, and attribute-level breakdowns."
)

# ---------------------- Summary KPIs ----------------------
st.header("Overall Summary of Claims and Severity Categories")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Number of Claims", f"{len(df):,}")
col2.metric("Critical Severity Claims", int((df["final_severity_label"] == "CRITICAL").sum()))
col3.metric("High Severity Claims", int((df["final_severity_label"] == "HIGH").sum()))
col4.metric("Medium Severity Claims", int((df["final_severity_label"] == "MEDIUM").sum()))
col5.metric("Low Severity Claims", int((df["final_severity_label"] == "LOW").sum()))

# ---------------------- Sidebar Filters ----------------------
st.sidebar.header("Filter the Dataset")

severity_options = sorted(df["final_severity_label"].unique(), reverse=True)
severity_filter = st.sidebar.multiselect(
    "Filter by Severity Category:",
    severity_options,
    default=severity_options
)

provider_filter = st.sidebar.text_input("Filter by Provider Identifier Contains:")
patient_filter = st.sidebar.text_input("Filter by Patient Identifier Contains:")

claimtype_filter = st.sidebar.multiselect(
    "Filter by Type of Claim (Optional):",
    options=sorted(df["ClaimType"].dropna().unique()) if "ClaimType" in df.columns else [],
    default=[]
)

state_filter = st.sidebar.multiselect(
    "Filter by State (Optional):",
    options=sorted(df["State"].dropna().unique()) if "State" in df.columns else [],
    default=[]
)

# ---------------------- Apply Filters ----------------------
df_filtered = df[df["final_severity_label"].isin(severity_filter)]

if provider_filter:
    df_filtered = df_filtered[df_filtered["ProviderID"].astype(str).str.contains(provider_filter, na=False)]

if patient_filter:
    df_filtered = df_filtered[df_filtered["PatientID"].astype(str).str.contains(patient_filter, na=False)]

if claimtype_filter:
    df_filtered = df_filtered[df_filtered["ClaimType"].isin(claimtype_filter)]

if state_filter:
    df_filtered = df_filtered[df_filtered["State"].isin(state_filter)]

# ---------------------- Data Quality Issues ----------------------
st.header("Overview of Data Quality Issues Identified in the Dataset")

dq_flag_cols = [c for c in df.columns if (
    c.startswith("bad_") or 
    c.startswith("flag_") or
    ("dup" in c.lower()) or
    c in ["missing_key", "dq_score", "dup_full_row", "dup_claimid_flag", "amount_suspect_flag"]
)]
dq_flag_cols = [c for c in dq_flag_cols if c in df.columns]

if dq_flag_cols:
    dq_summary = df[dq_flag_cols].sum().sort_values(ascending=False).reset_index()
    dq_summary.columns = ["Data Quality Issue", "Count"]

    fig_dq = px.bar(
        dq_summary,
        x="Data Quality Issue",
        y="Count",
        title="Count of Each Data Quality Issue",
        text_auto=True
    )
    st.plotly_chart(fig_dq, use_container_width=True)
    st.dataframe(dq_summary, use_container_width=True)
else:
    st.info("No data quality issue columns were detected.")

# ---------------------- Machine Learning Score Distributions ----------------------
st.header("Distribution of Machine Learning Based Anomaly Scores")

score_cols = [c for c in ["if_score_norm", "lof_score_norm", "hbos_score_norm", "combined_score"] if c in df.columns]

if score_cols:
    for col in score_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=60,
            title=f"Distribution of {col.replace('_',' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No machine learning score columns found.")

# ---------------------- ML vs DQ Scatterplot ----------------------
st.header("Comparison Between Machine Learning Score and Data Quality Score")

if "combined_score" in df.columns and "dq_score" in df.columns:
    fig_scatter = px.scatter(
        df,
        x="combined_score",
        y="dq_score",
        color="final_severity_label",
        hover_data=["ClaimID", "ProviderID", "PatientID", "ClaimAmount"],
        title="Relationship Between Machine Learning Score and Data Quality Score"
    )
    fig_scatter.update_traces(marker=dict(size=7, opacity=0.6))
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Either the combined score or the data quality score is missing.")

# ---------------------- Severity Distribution ----------------------
st.header("Distribution of Final Severity Labels Across All Claims")

sev_counts = df["final_severity_label"].value_counts().reset_index()
sev_counts.columns = ["Severity Category", "Count"]

fig_sev = px.bar(
    sev_counts,
    x="Severity Category",
    y="Count",
    color="Severity Category",
    title="Number of Claims in Each Severity Category",
    text_auto=True
)
st.plotly_chart(fig_sev, use_container_width=True)

if "final_severity_score" in df.columns:
    fig_score = px.histogram(
        df,
        x="final_severity_score",
        nbins=50,
        title="Distribution of Final Combined Severity Score"
    )
    st.plotly_chart(fig_score, use_container_width=True)

# ---------------------- Provider Hotspots ----------------------
st.header("Providers with the Highest Number of Severe Anomalies")

hot = df[df["final_severity_label"].isin(["CRITICAL", "HIGH"])]

if "ProviderID" in hot.columns:
    prov_hotspots = hot.groupby("ProviderID").size().reset_index(name="Anomaly Count")
    prov_hotspots = prov_hotspots.sort_values("Anomaly Count", ascending=False)

    fig_hot = px.bar(
        prov_hotspots.head(20),
        x="ProviderID",
        y="Anomaly Count",
        title="Top Providers With the Most Severe Anomalies"
    )
    st.plotly_chart(fig_hot, use_container_width=True)
else:
    st.info("Provider identifier column is missing.")

# ---------------------- Categorical Breakdown ----------------------
st.header("Breakdown of Anomalies Across Different Categorical Attributes")

cat_cols = ["ClaimType", "State", "PatientGender", "ProviderSpecialty"]

for col in cat_cols:
    if col in df.columns:
        temp = df.groupby([col, "final_severity_label"]).size().reset_index(name="Count")
        fig = px.bar(
            temp,
            x=col,
            y="Count",
            color="final_severity_label",
            title=f"Distribution of Anomalies by {col}"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Claim Amount Distribution ----------------------
st.header("Distribution of Claim Amounts Across Severity Categories")

if "ClaimAmount" in df.columns:
    fig_amt = px.histogram(
        df,
        x="ClaimAmount",
        color="final_severity_label",
        nbins=80,
        title="Variation of Claim Amounts Across Different Severity Levels"
    )
    st.plotly_chart(fig_amt, use_container_width=True)
else:
    st.info("Claim amount column not found in the dataset.")

# ---------------------- Top Anomalies Table ----------------------
st.header("Highest Severity Claims (Sorted by Severity Score)")

top_k = st.slider("Number of top claims to display:", 5, 100, 20) 
top = df.sort_values("final_severity_score", ascending=False).head(top_k)
st.dataframe(top, use_container_width=True)

# ---------------------- Claim Drilldown ----------------------
st.header("Detailed View of an Individual Claim")

claim_search = st.text_input("Enter a Claim ID to view its complete details:")
if claim_search:
    found = df[df["ClaimID"].astype(str) == str(claim_search)]
    if found.empty:
        st.warning("No claim found with the provided Claim ID.")
    else:
        st.success("Claim details found:")
        st.dataframe(found.transpose(), use_container_width=True)
