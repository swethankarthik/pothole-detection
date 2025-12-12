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

# -------------------------------------------------
# GLOBAL LIGHT UI THEME (OPTION B)
# -------------------------------------------------
st.markdown("""
<style>

/* === Clean background === */
body, [data-testid="stAppViewContainer"] {
    background-color: #F5F7FA;
}

/* Disable sidebar */
[data-testid="stSidebar"] {display: none;}

/* Typography */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-weight: 700 !important;
    color: #222 !important;
}

/* Buttons */
.stButton>button {
    background-color: #4A90E2 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}

/* Cards */
.report-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e3e6eb;
    margin-bottom: 15px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.06);
}

/* Severity badges */
.sev-large {color:#E63946; font-weight:700;}
.sev-medium {color:#F4A261; font-weight:700;}
.sev-small {color:#2A9D8F; font-weight:700;}
.sev-none {color:#6C757D; font-weight:700;}

/* Table styling */
thead tr th {
    background-color: #EFEFF6 !important;
    font-weight: 600 !important;
}

/* Hide inference dropdown completely */
div[data-baseweb="select"] {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("Pothole Detection & Reporting System")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = load_model(MODEL_PATH)
if model:
    st.success("YOLO model loaded successfully!")
else:
    st.warning("Model failed to load — using dummy mode.")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_user, tab_admin = st.tabs(["User Panel", "Admin Dashboard"])

# =============================================================================
# USER PANEL
# =============================================================================
with tab_user:

    st.header("Upload Image & Detect Potholes")

    uploaded_file = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if "current_image" not in st.session_state:
            st.session_state["current_image"] = None

        # Hidden — fixed inference method
        method = "tuned"

        if st.button("Run Detection"):
            with st.spinner("Detecting potholes..."):
                t0 = time.time()

                if method == "default":
                    detections = run_inference(image, model, conf_thresh=0.4)

                elif method == "tuned":
                    detections = run_inference_tuned(
                        image, model, conf_thresh=0.28, imgsz=1280, augment=True
                    )

                elif method == "tta":
                    detections = tta_inference(image, model, conf_thresh=0.25, imgsz=1280)

                else:
                    detections = tile_inference(
                        image, model, tile_size=768, overlap=0.25,
                        conf_thresh=0.25, imgsz=1280
                    )

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

    # Display detection results
    if (
        "results" in st.session_state
        and uploaded_file
        and st.session_state["current_image"] == uploaded_file.name
    ):
        r = st.session_state["results"]

        st.success(
            f"Detected {r['count']} potholes | Severity: {r['severity_label']} | "
            f"Area: {r['severity_pct']:.2f}% | Avg Conf: {r['avg_conf']:.2f} | "
            f"{r['infer_time']:.2f}s"
        )

        st.image(r["processed_image"], caption="Processed Output", use_column_width=True)

        with st.expander("Detection Details"):
            st.write(r["detections"])

        st.subheader("Save Report")

        with st.form("save"):
            col1, col2 = st.columns(2)
            latitude = col1.text_input("Latitude")
            longitude = col2.text_input("Longitude")
            notes = st.text_area("Notes (optional)")

            if st.form_submit_button("Save Report"):
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
                st.success(f"Report saved successfully — City: {city}")

# =============================================================================
# ADMIN DASHBOARD
# =============================================================================
with tab_admin:

    st.header("Admin Dashboard")

    if not os.path.exists(DATA_CSV):
        st.info("No reports yet.")
    else:

        df = pd.read_csv(DATA_CSV)

        # Fix missing columns
        expected = {
            "id": "", "timestamp": "", "city": "Unknown",
            "original_image": "", "processed_image": "",
            "latitude": "", "longitude": "",
            "pothole_count": 0, "severity": "none",
            "severity_pct": 0.0, "avg_confidence": 0.0, "notes": ""
        }

        for col, default in expected.items():
            if col not in df.columns:
                df[col] = default

        df["severity_pct"] = pd.to_numeric(df["severity_pct"], errors="coerce").fillna(0)
        df["avg_confidence"] = pd.to_numeric(df["avg_confidence"], errors="coerce").fillna(0)
        df["pothole_count"] = pd.to_numeric(df["pothole_count"], errors="coerce").fillna(0).astype(int)

        severity_order = {"large": 3, "medium": 2, "small": 1, "none": 0}
        df["severity_rank"] = df["severity"].map(severity_order)
        df = df.sort_values(["severity_rank", "pothole_count"], ascending=[False, False])

        # CSV Download
        st.subheader("All Reports")
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "pothole_reports.csv",
            "text/csv"
        )

        st.dataframe(
            df[["id", "timestamp", "city", "pothole_count",
                "severity", "severity_pct", "avg_confidence",
                "latitude", "longitude", "notes"]],
            use_container_width=True
        )

        # Cards
        st.subheader("Readable Report Cards")

        for _, row in df.iterrows():
            st.markdown(
                f"""
                <div class="report-card">
                    <div style="display:flex; justify-content:space-between;">
                        <span><b>ID:</b> {row['id']} • <b>City:</b> {row['city']}</span>
                        <span class="sev-{row['severity']}">{row['severity'].upper()}</span>
                    </div>

                    <div><b>Potholes:</b> {row['pothole_count']} • 
                        <b>Area%:</b> {row['severity_pct']:.2f}% • 
                        <b>Conf:</b> {row['avg_confidence']}
                    </div>

                    <div><b>Location:</b> {row['latitude']}, {row['longitude']}</div>
                    <div><b>Notes:</b> {row['notes'] or "—"}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Map
        st.subheader("Map View")

        df_valid = df[(df["latitude"] != "") & (df["longitude"] != "")]

        if len(df_valid):
            import folium
            from streamlit_folium import folium_static

            df_valid["latitude"] = df_valid["latitude"].astype(float)
            df_valid["longitude"] = df_valid["longitude"].astype(float)

            m = folium.Map(
                location=[df_valid["latitude"].mean(), df_valid["longitude"].mean()],
                zoom_start=12
            )

            for _, row in df_valid.iterrows():
                folium.Marker(
                    [row["latitude"], row["longitude"]],
                    popup=f"{row['city']}<br>Severity: {row['severity']}<br>Potholes: {row['pothole_count']}"
                ).add_to(m)

            folium_static(m, width=700, height=500)

        else:
            st.info("No GPS entries yet.")
