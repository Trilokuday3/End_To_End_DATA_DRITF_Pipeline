import streamlit as st
import glob
import os

st.title("Evidently Drift Dashboard Viewer")

dashboard_files = glob.glob("artifacts/drift_report_*.html")

if not dashboard_files:
    st.error("No drift dashboard HTMLs found in `artifacts/`. Run training script first.")
else:
    selection = st.selectbox("Select a Drift Report", dashboard_files)
    with open(selection, 'r', encoding='utf-8') as f:
        html = f.read()
    st.components.v1.html(html, height=900, scrolling=True)
    st.download_button("Download this dashboard", html, os.path.basename(selection), mime="text/html")