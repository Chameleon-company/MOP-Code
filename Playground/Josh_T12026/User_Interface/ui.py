import streamlit as st 
import requests

#PAGECONFIG
st.set_page_config(
    page_title="Smart Streetlight Fault Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

#send images to FastAPI
def backend_detect(files):
    url = "http://localhost:8000/detect"
    response = requests.post(url, files = files)
    return response.json()

#send detection results to FastAPI
def backend_report(detection_data):
    url = "http://localhost:8000/report"
    response = requests.post(url, json = detection_data)
    return response.json()

#SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Detection", "Reports"])

#HOMEPAGE 
if page == "Homepage": 
    st.title(":blue[Smart Streetlight Fault Detection] :bulb:", text_alignment = "center")
    
    st.markdown("""
                This project utilises a vision-based system to analyse provided nighttime images and detects faulty streetlights, specifically: 
                - Streetlights that are either 
                    - **not functioning** 
                    - **flickering** 
                    - **producing a weak illumination**
                
                Following this, a maintenence alert will be generated. 
                
                Please proceed to the detection page to get started. """)

#DETECTIONPAGE
elif page == "Detection": 
    st.title("Streetlight Analysis")
    
    #hold the result for backend use 
    stored_result = None
    
    #COLUMNS
    col1, col2 = st.columns([1, 2])
    
    with col1: 
    #file uploader
        uploaded_files = st.file_uploader("Upload image(s)", accept_multiple_files = True, type = ["jpg", "png"])

    with col2: 
    #loop through uploaded files, display preview, and add a button 
        if uploaded_files: 
            st.header("Preview uploaded images")
            for file in uploaded_files: 
                st.image(file, width = "content")
                st.markdown("---")
            
            #button for analysis 
            if st.button("Analyse image(s)", type = "primary"):
                with st.spinner("Analysing images..."):
                    #prep files 
                    file_data = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for f in uploaded_files
                    ]
                    
                    #send to backend
                    stored_result = backend_detect(file_data)
                    
                    st.session_state["analysis_result"] = stored_result 
                    
                st.success("Analysis complete!")
        else:
            st.info("No uploaded images.")

    #DIVIDE PAGE 
    st.markdown("---")

    st.header("Analysis Results")
    if "analysis_result" in st.session_state:
        st.json(st.session_state["analysis_result"])
    else:
        st.info("No analysis results")

#REPORTSPAGE
elif page == "Reports": 
    st.title("Maintenance Report")
    
    if st.button("Generate report", type = "primary"): 
        if "analysis_result" not in st.session_state: 
            st.warning("No results available. Please run detection analysis first")
        else: 
            with st.spinner("Generating report..."):
                detection_data = st.session_state["analysis_result"]
                
                #send to backend 
                report = backend_report(detection_data)
                
            st.success("Report generated!")
            st.json(report)