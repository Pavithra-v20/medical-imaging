import streamlit as st
import requests
import json
import time
import pydicom
import numpy as np
import io

# --- Configuration ---
API_URL = "http://127.0.0.1:8080"

st.set_page_config(
    page_title="CT Disease Agent UI",
    page_icon="🏥",
    layout="wide"
)

# --- DICOM Viewer Function ---
def dicom_to_image(file_bytes, window_center=40, window_width=400):
    """
    Parses DICOM bytes and converts pixel data to a displayable image.
    Applies a windowing function to map raw pixel values to grayscale.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        
        # Get pixel array
        pixel_array = ds.pixel_array.astype(float)
        
        # Apply Rescale Slope and Intercept if they exist
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
        # Apply windowing
        lower_bound = window_center - (window_width / 2)
        upper_bound = window_center + (window_width / 2)
        
        pixel_array[pixel_array < lower_bound] = lower_bound
        pixel_array[pixel_array > upper_bound] = upper_bound
        
        # Normalize to 0-255
        image_2d = (pixel_array - lower_bound) / (upper_bound - lower_bound)
        image_2d_scaled = (image_2d * 255.0).astype(np.uint8)
        
        return image_2d_scaled
    except Exception as e:
        st.error(f"Failed to parse DICOM file: {e}")
        return None

# --- UI ---
st.title("🏥 CT Disease Agent")
st.markdown("""
Upload a DICOM CT scan to trigger the end-to-end AI pipeline. 
The agent will perform **Anonymization**, **3D Segmentation**, and **Disease Prediction**.
""")

# --- Sidebar: Inputs ---
st.sidebar.header("Scan Metadata")
tech_id = st.sidebar.text_input("Technician ID", value="tech-01")
patient_id = st.sidebar.text_input("Patient ID (Anonymized)", value="patient-42")
physician_id = st.sidebar.text_input("Reviewing Physician", value="dr-jones")

st.sidebar.divider()
st.sidebar.info("The agent uses NVIDIA VISTA-3D for segmentation and CT-CHAT for prediction.")

# --- Main: File Upload & Viewer ---
uploaded_file = st.file_uploader("Choose a DICOM file (.dcm)", type=["dcm"])

if uploaded_file is not None:
    # Read bytes once
    dicom_bytes = uploaded_file.getvalue()
    
    # Display the DICOM image
    st.subheader("DICOM Preview")
    image_to_display = dicom_to_image(dicom_bytes)
    
    if image_to_display is not None:
        st.image(image_to_display, caption=f"DICOM Preview for {uploaded_file.name}", use_column_width=True)

    st.divider()

    if st.button("🚀 Run AI Pipeline"):
        with st.status("Processing scan...", expanded=True) as status:
            st.write("📤 Uploading to agent...")
            
            # Prepare request
            files = {"file": (uploaded_file.name, dicom_bytes, "application/dicom")}
            data = {
                "technician_id": tech_id,
                "patient_id": patient_id,
                "physician_id": physician_id
            }
            
            try:
                start_time = time.time()
                response = requests.post(f"{API_URL}/predict", data=data, files=files, timeout=300) # Added timeout
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    status.update(label=f"Analysis Complete ({duration:.1f}s)", state="complete", expanded=False)
                    result = response.json()
                    
                    # --- Results Display ---
                    st.divider()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📝 Summary")
                        st.info(result["summary"])
                        
                        st.subheader("📋 Radiology Report")
                        st.text_area("Findings & Impression", value=result["report"], height=300)
                    
                    with col2:
                        st.subheader("🔬 AI Prediction")
                        pred = result["prediction"]
                        
                        # Metrics
                        st.metric("Disease Label", pred.get("disease_label", "N/A").title())
                        st.metric("Confidence", f"{pred.get('confidence', 0)*100:.1f}%")
                        
                        # Review Requirement
                        if result["rev_req"]:
                            st.warning("⚠️ Physician Review Required")
                        else:
                            st.success("✅ Confidence Threshold Met")
                            
                        with st.expander("Clinical Reasoning"):
                            st.write(pred.get("reasoning", "No reasoning provided."))
                        
                        st.caption(f"Session ID: {result['session_id']}")
                        st.caption(f"Model: {pred.get('model_used', 'N/A')}")

                else:
                    status.update(label="Error in Pipeline", state="error")
                    st.error(f"Agent returned error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                status.update(label="Connection Failed", state="error")
                st.error(f"Could not connect to Agent at {API_URL}. Is the FastAPI server running? Details: {e}")

else:
    st.info("Please upload a DICOM file to begin.")

# --- Footer ---
st.divider()
st.caption("ValueHealth Intern-eBV | CT Disease Agent Prototype")

