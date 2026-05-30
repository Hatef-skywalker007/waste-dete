from pathlib import Path
import tempfile
from PIL import Image
import cv2
import numpy as np
import streamlit as st
import helper
import settings

st.set_page_config(page_title="Waste Detection(DEMO)")

st.sidebar.title("Detection Console (TEST)")
model_path = Path(settings.DETECTION_MODEL)

st.title("Intelligent waste segregation system")

st.markdown(
    """
    <style>
        .stRecyclable {
            background-color: rgba(233,192,78,255);
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            margin-top: 0 !important;
            font-size:18px !important;
        }
        .stNonRecyclable {
            background-color: rgba(94,128,173,255);
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            margin-top: 0 !important;
            font-size:18px !important;
        }
        .stHazardous {
            background-color: rgba(194,84,85,255);
            padding: 1rem 0.75rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            margin-top: 0 !important;
            font-size:18px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

# -------------------------------------------------------------------
# Helper functions for detection on uploaded files
# -------------------------------------------------------------------
def detect_image(model, image: Image.Image):
    """Run YOLO inference on a PIL image and return annotated image + detection summary."""
    results = model.predict(image, conf=0.5)  # adjust confidence as needed
    annotated = results[0].plot()  # BGR numpy array
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    # Count classes (adjust class names based on your model)
    class_counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    return annotated_rgb, class_counts

def detect_video(model, video_bytes, output_path="processed_video.mp4", process_every_n=2):
    """Process an uploaded MP4 video, write annotated output, and return output path."""
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(video_bytes)
        input_video_path = tmp_input.name

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process every Nth frame for speed (optional)
        if frame_count % process_every_n == 0:
            results = model.predict(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame  # skip detection for this frame
        out.write(annotated_frame)
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    # Clean up input temp file
    Path(input_video_path).unlink()
    return output_path

def display_summary(class_counts):
    """Show detection summary using custom CSS classes."""
    waste_mapping = {
        "recyclable": "Recyclable",
        "non-recyclable": "NonRecyclable",
        "hazardous": "Hazardous"
    }
    for class_name, count in class_counts.items():
        # Map class name to CSS class (case‑insensitive)
        css_class = "stRecyclable"
        lower_name = class_name.lower()
        if "recycl" in lower_name:
            css_class = "stRecyclable"
        elif "non" in lower_name or "organic" in lower_name:
            css_class = "stNonRecyclable"
        elif "hazard" in lower_name or "toxic" in lower_name:
            css_class = "stHazardous"
        st.markdown(
            f'<div class="{css_class}">{class_name}: {count} item(s)</div>',
            unsafe_allow_html=True,
        )

# Create tabs: Live Webcam (original) + Upload Media (new)

tab1, tab2 = st.tabs(["📷 Live Webcam", "📂 Upload Image / Video"])

with tab1:
    st.write("Start detecting objects in the webcam stream by clicking the button below. To stop detection, click the stop button in the top right corner of the webcam stream.")
    helper.play_webcam(model)

with tab2:
    st.subheader("Detect from an image or video file")
    uploaded_file = st.file_uploader(
        "Choose an image (jpg, jpeg, png) or video (mp4)",
        type=["jpg", "jpeg", "png", "mp4"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type
        if "image" in file_type:
            # Process image
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("Detecting objects..."):
                annotated_img, class_counts = detect_image(model, image)
            st.image(annotated_img, caption="Detection Result", use_container_width=True)
            st.subheader("Detection Summary")
            display_summary(class_counts)

        elif "video" in file_type:
            # Process video
            video_bytes = uploaded_file.read()
            with st.spinner("Processing video (this may take a while)..."):
                # Use a temporary file for output video
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                    out_path = tmp_out.name
                detect_video(model, video_bytes, output_path=out_path, process_every_n=3)
                st.success("Video processed!")
                st.video(out_path)
                st.info("Note: Object counts per frame are not aggregated automatically. Open the processed video to see detections.")
                # Optionally, we could run a summary pass, but this keeps it simple.
        else:
            st.error("Unsupported file type. Please upload an image or MP4 video.")

st.sidebar.markdown(
    "This is a demo of the waste detection model MADE BY hatef_jani.",
    unsafe_allow_html=True,
)