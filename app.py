import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

# App Configuration
st.set_page_config(
    page_title="Dental Implant Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9f7ef;
        background-image: linear-gradient(180deg, #e9f7ef 0%, #d4edf7 100%);
    }
    h1 {
        color: #2b5876;
        border-bottom: 2px solid #4a90e2;
        padding-bottom: 10px;
    }
    .stButton>button {
        background-color: #4a90e2;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2b5876;
        transform: scale(1.05);
    }
    .stFileUploader>div>div>div>div {
        color: #2b5876;
    }
    .stProgress > div > div > div {
        background-image: linear-gradient(90deg, #4a90e2 0%, #2b5876 100%);
    }
    .stAlert {
        border-left: 5px solid #4a90e2;
    }
</style>
""", unsafe_allow_html=True)

# Cache the YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# Image Processing Functions
def convert_to_grayscale(image):
    """Convert PIL image to grayscale using OpenCV"""
    image_np = np.array(image)  # Convert to NumPy array
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(gray_image)  # Convert back to PIL image

def run_detection(_model, image, conf_threshold):
    """Run YOLO object detection and draw bounding boxes"""
    image_np = np.array(image.convert('RGB'))  # Convert PIL image to NumPy
    results = _model(image_np, conf=conf_threshold)  

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        label = f"Implant: {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Put label text
        cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return Image.fromarray(image_np), results[0]

# Main Application
def main():
    st.title("🦷 Advanced Dental Implant Detection")
    st.markdown("""
    **AI-powered detection system** for identifying dental implants in radiographic images with YOLOv11m architecture.
    """)
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        st.header("Model Information")
        st.code(f"Model: YOLOv11m\nWeights: best.pt", language="python")
        
        st.markdown("---")
        st.header("Statistics")
        if 'processed_count' not in st.session_state:
            st.session_state.processed_count = 0
        st.metric("Images Processed", st.session_state.processed_count)

    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload Dental Radiographs (Images)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=False,
        key="file_uploader"
    )

    if uploaded_file and model:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            original_img = Image.open(uploaded_file)
            st.image(original_img, use_container_width=True)
            
            # Convert to grayscale
            gray_img = convert_to_grayscale(original_img)
            st.image(gray_img, caption="Grayscale Image", use_container_width=True)
        
        with col2:
            st.subheader("Detection Results")
            with st.spinner("Detecting implants..."):
                detected_img, results = run_detection(model, gray_img, conf_threshold)  # Using grayscale image
                st.image(detected_img, use_container_width=True)
                
                # Display detection metrics
                if hasattr(results, 'boxes'):
                    num_detections = len(results.boxes)
                    st.success(f"Detected {num_detections} implants")
                    st.session_state.processed_count += 1
                    
                    # Confidence distribution
                    if num_detections > 0:
                        confidences = results.boxes.conf.cpu().numpy()
                        st.metric("Average Confidence", f"{np.mean(confidences):.2f}")
                
                # Download results
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    detected_img.save(tmp.name, quality=95)
                    btn = st.download_button(
                        label="Download Result",
                        data=open(tmp.name, 'rb'),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
                os.unlink(tmp.name)

if __name__ == "__main__":
    main()
