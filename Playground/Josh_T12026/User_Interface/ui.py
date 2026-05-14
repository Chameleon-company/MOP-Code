import streamlit as st 
import requests
import base64

#PAGECONFIG
st.set_page_config(
    page_title = "Smart Streetlight Fault Detection",
    layout = "centered",
    initial_sidebar_state = "expanded"
)

#send images to FastAPI
def backend_detect(files):
    try: 
        url = "http://localhost:8000/detect"
        response = requests.post(url, files = files)
        return response.json()
    except: 
        return {"error": "Backend unavailable"}

#send detection results to FastAPI
def backend_report(detection_data):
    try: 
        url = "http://localhost:8000/report"
        response = requests.post(url, json = detection_data)
        return response.json()
    except: 
        return {"error": "Backend unavailable"}

def backend_get_reports():
    try:
        response = requests.get("http://localhost:8000/reports")
        return response.json()
    except:
        return {"reports": []}

#SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Detection", "Reports", "Report History", "About"])

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
                st.image(file, width="stretch")
                st.markdown("---")
            
            #button for analysis 
            if st.button("Analyse image(s)", type = "primary"):
                with st.spinner("Analysing images..."):
                    #prep files 
                    file_data = [
                        ("files", (file.name, file.getvalue(), file.type))
                        for file in uploaded_files
                    ]
                    
                    #send to backend
                    result = backend_detect(file_data)
                    if isinstance(result, dict) and "results" in result:
                        st.session_state["analysis_result"] = result
                        st.success("Analysis complete!")
                    else:
                        st.error("Backend returned invalid response")
                        st.json(result)
        else:
            st.info("No uploaded images.")

    #DIVIDE PAGE 
    st.markdown("---")

    #analysis results section 
    st.header("Analysis Results")
    if "analysis_result" in st.session_state:
        result_data = st.session_state["analysis_result"]

        if isinstance(result_data, dict) and "results" in result_data:
            for result in result_data["results"]:

                analysis = result["analysis"]
                st.subheader(result["image"])

                #if an image exists 
                if "uploaded_img" in analysis:
                    image_bytes = base64.b64decode(
                        analysis["uploaded_img"]
                    )
                    st.image(image_bytes)

                st.write(f"Streetlights: {analysis['streetlight_count']}")
                st.write(f"On: {analysis['on']}")
                st.write(f"Dim: {analysis['dim']}")
                st.write(f"Off: {analysis['off']}")

                st.json(analysis["details"])

                st.markdown("---")
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
                detection_data = st.session_state['analysis_result']
                
                #send to backend 
                report = backend_report(detection_data)
                
            if isinstance(report, dict) and "results" in report:
                st.success("Report generated!")

                for item in report["results"]: 
                    st.subheader(item["image"])
                    st.write(item["report"])
                    st.markdown("---")
            else:
                st.error("Report failed or invalid response from backend")
                st.json(report)
    
#REPORTHISTORY
elif page == "Report History": 
    st.title("Report History")
    data = backend_get_reports() 
    reports = data.get("reports", [])
    
    if not reports: 
        st.warning("No reports found. ")
    else: 
        for report in reports:
            st.subheader(f"Report: {report['report_id']}")

            for item in report["results"]:

                image_url = f"http://localhost:8000/uploads/{report['report_id']}/{item['image']}"

                st.image(image_url)

                st.json(item["analysis"])

                if "report" in item:
                    st.success(item["report"])
                else:
                    st.warning("No LLM report available")


    
#ABOUTPAGE
elif page == "About": 
    st.title(":blue[About Us]", text_alignment = "center")
    
    st.markdown("""
            Our project team is composed of the following members: 
            
            **Syed Hamiz Hassan** 
            - Computer Vision Modelling 
            
                
            **Savith Mundukotuwa** 
            - Data Preparation 
            
            
            **Josh Wong** 
            - User Interface and System Integration
            
            
            **Luke Kankannamge Don** 
            - LLM Reporting System 
            
            
            **Rahul Sheoran** 
            - Model Analysis
             
            """)