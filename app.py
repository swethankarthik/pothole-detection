import streamlit as st
from PIL import Image
import os
from datetime import datetime
import uuid
import pandas as pd
import time

from utils import (
    load_model,
    run_inference,
    run_inference_tuned,
    tta_inference,
    tile_inference,
    draw_boxes,
    compute_severity,
    save_report,
    get_city_name
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "models/best.pt"
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
DATA_CSV = "data/reports.csv"

for d in [UPLOADS_DIR, OUTPUTS_DIR, os.path.dirname(DATA_CSV)]:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------
# STREAMLIT SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Pothole Detection & Reporting",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("Pothole Detection & Reporting (Hackathon Demo)")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = load_model(MODEL_PATH)
if model:
    st.success("YOLO model loaded successfully!")
else:
    st.warning("Model failed to load. Dummy mode active.")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_user, tab_admin = st.tabs(["User Panel", "Admin Dashboard"])

# =============================================================================
# USER PANEL
# =============================================================================
with tab_user:
    st.header("Upload & Detect Potholes")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if "current_image" not in st.session_state:
            st.session_state["current_image"] = None

        method = st.selectbox(
            "Choose Inference Method:",
            ["default", "tuned", "tta", "tiled"],
            index=1
        )

        if st.button("Run Detection"):
            with st.spinner("Running Detection..."):
                t0 = time.time()

                if method == "default":
                    detections = run_inference(image, model, conf_thresh=0.4)
                elif method == "tuned":
                    detections = run_inference_tuned(image, model, conf_thresh=0.28, imgsz=1280, augment=True)
                elif method == "tta":
                    detections = tta_inference(image, model, conf_thresh=0.25, imgsz=1280)
                else:
                    detections = tile_inference(image, model, tile_size=768, overlap=0.25, conf_thresh=0.25, imgsz=1280)

                infer_time = time.time() - t0

                processed_image = draw_boxes(image, detections)

                uid = str(uuid.uuid4())[:8]
                orig_name = f"{uid}_orig.jpg"
                proc_name = f"{uid}_proc.jpg"

                image.save(os.path.join(UPLOADS_DIR, orig_name))
                processed_image.save(os.path.join(OUTPUTS_DIR, proc_name))

                count = len(detections)
                severity_label, severity_pct = compute_severity(image, detections)
                avg_conf = sum(d["conf"] for d in detections) / count if count else 0.0

                st.session_state["results"] = {
                    "detections": detections,
                    "processed_image": processed_image,
                    "uid": uid,
                    "orig_name": orig_name,
                    "proc_name": proc_name,
                    "count": count,
                    "severity_label": severity_label,
                    "severity_pct": severity_pct,
                    "avg_conf": avg_conf,
                    "infer_time": infer_time
                }
                st.session_state["current_image"] = uploaded_file.name

        # Show results
        if "results" in st.session_state and st.session_state["current_image"] == uploaded_file.name:
            r = st.session_state["results"]

            st.success(
                f"Detections: {r['count']} | Severity: {r['severity_label']} | "
                f"Area%: {r['severity_pct']:.2f}% | Avg Conf: {r['avg_conf']:.2f} | Time: {r['infer_time']:.2f}s"
            )

            st.image(r["processed_image"], caption="Processed Image", use_column_width=True)

            with st.expander("Raw Detection Data"):
                st.write(r["detections"])

            # Save report
            st.subheader("Save Report")
            with st.form("save_report"):
                col_lat, col_lon = st.columns(2)
                latitude = col_lat.text_input("Latitude")
                longitude = col_lon.text_input("Longitude")
                notes = st.text_area("Notes (optional)")

                save_btn = st.form_submit_button("Save Report")
                if save_btn:
                    city = get_city_name(latitude, longitude)

                    entry = {
                        "id": r["uid"],
                        "timestamp": datetime.utcnow().isoformat(),
                        "original_image": r["orig_name"],
                        "processed_image": r["proc_name"],
                        "latitude": latitude,
                        "longitude": longitude,
                        "city": city,
                        "pothole_count": r["count"],
                        "severity": r["severity_label"],
                        "severity_pct": r["severity_pct"],
                        "avg_confidence": round(r["avg_conf"], 3),
                        "notes": notes
                    }

                    save_report(DATA_CSV, entry)
                    st.success(f"Report saved! 📍 City detected as **{city}**")

# =============================================================================
# ADMIN DASHBOARD
# =============================================================================
with tab_admin:
    st.header("Admin Dashboard")

    if not os.path.exists(DATA_CSV):
        st.info("No reports found yet.")
    else:
        df = pd.read_csv(DATA_CSV)

        # Add missing columns if old CSV
        expected_cols = {
            "id": "",
            "timestamp": "",
            "city": "Unknown",
            "original_image": "",
            "processed_image": "",
            "latitude": "",
            "longitude": "",
            "pothole_count": 0,
            "severity": "none",
            "severity_pct": 0.0,
            "avg_confidence": 0.0,
            "notes": ""
        }
        for col, default in expected_cols.items():
            if col not in df.columns:
                df[col] = default

        # Convert types
        df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0).astype(int)
        df["severity_pct"] = pd.to_numeric(df["severity_pct"], errors="coerce").fillna(0)
        df["avg_confidence"] = pd.to_numeric(df["avg_confidence"], errors="coerce").fillna(0)

        # Sort by severity + pothole count
        severity_order = {"large": 3, "medium": 2, "small": 1, "none": 0}
        df["severity_rank"] = df["severity"].map(severity_order)
        df = df.sort_values(["severity_rank", "pothole_count"], ascending=[False, False])

        # =================================================
        # DOWNLOAD REPORT BUTTON
        # =================================================
        st.subheader("Pothole List (All Reports)")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_data,
            file_name="pothole_reports.csv",
            mime="text/csv"
        )

        # Table view
        df_display = df[[
            "id", "timestamp", "city", "pothole_count", "severity",
            "severity_pct", "avg_confidence", "latitude", "longitude", "notes"
        ]]
        st.dataframe(df_display, use_container_width=True)

        # =================================================
        # HUMAN‑FRIENDLY CARDS
        # =================================================
        st.markdown("### 📋 Human-Friendly Report View")
        for _, row in df.iterrows():
            st.markdown(
                f"""
                <div style="
                    padding:15px;
                    margin-bottom:12px;
                    border-radius:10px;
                    background-color:#1f1f1f;
                    border:1px solid #444;
                ">
                    <b>ID:</b> {row['id']} <br>
                    <b>Timestamp:</b> {row['timestamp']} <br>
                    <b>City:</b> {row['city']} <br>
                    <b>Pothole Count:</b> {row['pothole_count']} <br>
                    <b>Severity:</b> {row['severity']} <br>
                    <b>Area %:</b> {row['severity_pct']:.2f}% <br>
                    <b>Avg Confidence:</b> {row['avg_confidence']} <br>
                    <b>Location:</b> {row['latitude']}, {row['longitude']} <br>
                    <b>Notes:</b> {row['notes'] or "—"} <br>
                </div>
                """,
                unsafe_allow_html=True
            )

        # =================================================
        # RECENT IMAGES
        # =================================================
        st.subheader("Recent Processed Images")

        df["processed_image"] = df["processed_image"].astype(str)
        recent = df.head(5)

        for _, row in recent.iterrows():
            img_path = os.path.join(OUTPUTS_DIR, row["processed_image"])
            if os.path.exists(img_path):
                st.image(
                    img_path,
                    width=250,
                    caption=f"{row['timestamp']} | {row['city']} | "
                            f"{row['severity']} | {row['pothole_count']} potholes"
                )

        # =================================================
        # MAP VIEW
        # =================================================
        st.subheader("Map View (GPS Locations)")

        df_valid = df[(df["latitude"] != "") & (df["longitude"] != "")]

        if len(df_valid):
            import folium
            from streamlit_folium import folium_static

            try:
                df_valid["latitude"] = df_valid["latitude"].astype(float)
                df_valid["longitude"] = df_valid["longitude"].astype(float)
            except:
                st.warning("Some coordinates couldn't be parsed.")

            avg_lat = df_valid["latitude"].mean()
            avg_lon = df_valid["longitude"].mean()

            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

            for _, row in df_valid.iterrows():
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=f"{row['city']}<br>Severity: {row['severity']}<br>"
                          f"Potholes: {row['pothole_count']}",
                    tooltip=row["city"]
                ).add_to(m)

            folium_static(m, width=700, height=500)
        else:
            st.info("No valid GPS entries yet.")
