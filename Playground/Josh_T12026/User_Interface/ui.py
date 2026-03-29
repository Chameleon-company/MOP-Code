import streamlit as st 
from PIL import Image 
import requests

#PAGECONFIG
st.set_page_config(
    page_title="Smart Streetlight Fault Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

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
                    import time
                    time.sleep(2)  # simulate processing delay

                st.success("Analysis complete!")

    #DIVIDE PAGE 
    st.markdown("---")

    st.header("Analysis Results")
    st.write("This section will display:")
    st.markdown("""
    - Number of detected streetlights  
    - Number of faulty lights  
    """)

    st.success("Results placeholder")

#REPORTSPAGE
elif page == "Reports": 
    st.title("Maintenance Report")
    
    st.info("Report placeholder")
