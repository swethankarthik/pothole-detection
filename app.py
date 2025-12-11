import streamlit as st
from PIL import Image
import os
from datetime import datetime
import uuid
import pandas as pd

# Import helper functions from utils.py
from utils import (
    load_model,
    run_inference,
    draw_boxes,
    compute_severity,
    save_report
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "models/best.pt"   # your trained YOLO model
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
DATA_CSV = "data/reports.csv"

# Ensure folders exist
for d in [UPLOADS_DIR, OUTPUTS_DIR, os.path.dirname(DATA_CSV)]:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------
# STREAMLIT SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Pothole Detection & Reporting",
    layout="wide",
    initial_sidebar_state="collapsed"   # collapse sidebar
)

# ---- HIDE SIDEBAR COMPLETELY ----
hide_sidebar = """
<style>
[data-testid="stSidebar"] {display: none;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

st.title("Pothole Detection & Reporting (Hackathon Demo)")

# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------
model = load_model(MODEL_PATH)

if model is None:
    st.warning("Model not loaded. Dummy detection will be used.")
else:
    st.success("YOLO model loaded successfully!")

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
col1, col2 = st.columns([2, 1])

# -------------------------------------------------
# LEFT PANEL — UPLOAD + DETECTION
# -------------------------------------------------
with col1:
    st.header("Upload Road Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image Preview", use_column_width=True)

        if "current_image" not in st.session_state:
            st.session_state["current_image"] = None

        # Run Detection
        if st.button("Run Detection"):
            with st.spinner("Running YOLO model..."):
                detections = run_inference(image, model, 0.4)
                processed_image = draw_boxes(image, detections)

                uid = str(uuid.uuid4())[:8]
                orig_name = f"{uid}_orig.jpg"
                proc_name = f"{uid}_proc.jpg"

                image.save(os.path.join(UPLOADS_DIR, orig_name))
                processed_image.save(os.path.join(OUTPUTS_DIR, proc_name))

                count = len(detections)
                severity_label, severity_pct = compute_severity(image, detections)
                avg_conf = sum([d["conf"] for d in detections]) / count if count else 0.0

                st.session_state["results"] = {
                    "detections": detections,
                    "processed_image": processed_image,
                    "uid": uid,
                    "orig_name": orig_name,
                    "proc_name": proc_name,
                    "count": count,
                    "severity_label": severity_label,
                    "severity_pct": severity_pct,
                    "avg_conf": avg_conf
                }
                st.session_state["current_image"] = uploaded_file.name

        # Show Results
        if "results" in st.session_state and st.session_state["current_image"] == uploaded_file.name:
            r = st.session_state["results"]

            st.success(
                f"Detections: {r['count']} | Severity: {r['severity_label']} | Avg Conf: {r['avg_conf']:.2f}"
            )

            st.image(
                r["processed_image"],
                caption="Processed Image with Detections",
                use_column_width=True
            )

            with st.expander("Raw Detection Data"):
                st.write(r["detections"])

            # Save Report
            st.subheader("Save Report")

            with st.form("save_report"):
                col_lat, col_lon = st.columns(2)
                latitude = col_lat.text_input("Latitude")
                longitude = col_lon.text_input("Longitude")
                notes = st.text_area("Notes (optional)")
                submit = st.form_submit_button("Save Report")

                if submit:
                    entry = {
                        "id": r["uid"],
                        "timestamp": datetime.utcnow().isoformat(),
                        "original_image": r["orig_name"],
                        "processed_image": r["proc_name"],
                        "latitude": latitude,
                        "longitude": longitude,
                        "pothole_count": r["count"],
                        "severity": r["severity_label"],
                        "severity_pct": r["severity_pct"],
                        "avg_confidence": round(r["avg_conf"], 3),
                        "notes": notes
                    }

                    save_report(DATA_CSV, entry)
                    st.success("Report saved!")

# -------------------------------------------------
# RIGHT PANEL — ADMIN + MAP
# -------------------------------------------------
with col2:
    st.header("Admin / Reports")

    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)

        df["processed_image"] = df["processed_image"].astype(str)

        st.dataframe(df.tail(10))

        st.subheader("Recent Processed Images")
        recent = df.tail(5).sort_values("timestamp", ascending=False)

        for _, row in recent.iterrows():
            img_path = os.path.join(OUTPUTS_DIR, row["processed_image"])

            if os.path.exists(img_path):
                st.image(
                    img_path,
                    width=250,
                    caption=f"{row['timestamp']} | {row['severity']} | {row['pothole_count']} potholes"
                )

    else:
        st.info("No reports yet. Save one after detection.")

    # ---------------- MAP VIEW ----------------
    st.subheader("Map View (GPS Locations)")

    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        df_valid = df[(df["latitude"] != "") & (df["longitude"] != "")]

        if len(df_valid) > 0:
            import folium
            from streamlit_folium import folium_static

            df_valid["latitude"] = df_valid["latitude"].astype(float)
            df_valid["longitude"] = df_valid["longitude"].astype(float)

            # Center map on ALL points (fixed!)
            avg_lat = df_valid["latitude"].mean()
            avg_lon = df_valid["longitude"].mean()

            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

            for _, row in df_valid.iterrows():
                popup_text = f"Severity: {row['severity']}<br>Potholes: {row['pothole_count']}"
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=popup_text,
                    tooltip="Pothole Report"
                ).add_to(m)

            folium_static(m, width=350, height=500)

        else:
            st.info("No valid GPS entries yet.")
