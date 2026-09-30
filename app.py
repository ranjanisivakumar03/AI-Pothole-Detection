import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="AI Road Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ AI-Based Road Pothole Detection")
st.write("Upload a road image and the AI model will detect potholes.")

@st.cache_resource
def load_model():
    return YOLO(
        "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
    )

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Potholes"):

        with st.spinner("Detecting potholes..."):

            results = model.predict(
                image,
                conf=0.40
            )

        result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

        pothole_count = len(results[0].boxes)

        st.success(
            f"Detection completed! Potholes detected: {pothole_count}"
        )