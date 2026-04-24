from LLM_pipeline.llm import report_generation


print("reportGenerationHelpers loaded")

def fetchSingleRow(supabase, id):
    try:
        print("Called")
        row = supabase.table("crack_reports").select("*").eq("id", id).execute()
        if row.data is None:
            return None
        return row.data[0]     
    except Exception as e:
        raise Exception
    
    
def uploadReport(supabase, report):
    print(f"Report = {report}")
    try:
        data = supabase.table('crack_reports').insert({
            'imageid': report["image_id"],
            'severity': report["severity"],
            'numcracks': report["num_crack_regions"],
            'crackarearatio': report["largest_crack_area_ratio"],
            'estimatedcracklength': report["largest_crack_length"],
            'damagelevel': report["damage_level"],
            'reportstatus': report["report_status"],
            'riskassessment': report["risk_assessment"],
            'repairactions': report["repair_actions"],
            'inspectionschedule': report["inspection_schedule"]            
        }).execute()
    except Exception as e:
        raise

    
def convertReport(oldReport):
    report = {
        "id": oldReport["id"],
        "image_id": oldReport["imageid"],
        "severity": oldReport["severity"],
        "num_crack_regions": oldReport["numcracks"],
        "largest_crack_area_ratio": oldReport["crackarearatio"],
        "largest_crack_length": oldReport["estimatedcracklength"],
        "damage_level": oldReport["damagelevel"]
    }
    print(f"Old report = {oldReport}")
    print(f"New report = {report}")
    return report


def changeRowStatus(supabase, id, status):
    if status != "Pending" and status != "None":
        print("status must be Pending or None")
        raise Exception
    try:
        supabase.table("crack_reports").update({"reportstatus": status,}).eq("id", id).execute()
        print(f"Called change row for {id} to {status}")
    except Exception as e:
        print(f"Error updating row {id} to {status}. Error: {e}")


def generateAiReport(report):
    try:
        llm_result = report_generation(report)
        sections = llm_result['sections']    
        report["reportStatus"] = "Generated"    
        report["risk_assessment"] = sections['risk_assessment']['reasoning']   
        report["repair_actions"] = sections['repair_actions']['reasoning']   
        report["inspection_schedule"] = sections['inspection_schedule']['reasoning']  
        
        return report
    except ConnectionError as e:
        raise Exception(f"Connection Error {e}")
    except TimeoutError as e:
        raise Exception(f"Timeout Error: {e}")
    except Exception as e:
        raise Exception(f"General Error: {e}")
    
    

def updateRowWithAiReport(supabase, report):
    try:
        supabase.table("crack_reports").update({
            "reportstatus": report["reportStatus"],
            "riskassessment": report["risk_assessment"],
            "repairactions": report["repair_actions"],
            "inspectionschedule": report["inspection_schedule"]
        }).eq("id", report["id"]).execute()
    
    except Exception as e:
        raise Exception(f"Failed to update row with AI report: {e}")
