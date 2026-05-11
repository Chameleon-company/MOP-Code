from LLM_pipeline.llm import report_generation


print("reportGenerationHelpers loaded")

# ── Fetch Single Row ────────────────────
def fetchSingleRow(supabase, id):
    try:
        print("Called")
        row = supabase.table("crack_reports").select("*").eq("id", id).execute()
        if row.data is None:
            return None
        return row.data[0]
    except Exception as e:
        raise Exception

# ── Upload Report ────────────────────
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
            'inspectionschedule': report["inspection_schedule"],
            'imageurl': report["image_url"],
            'crackmaskurl': report["mask_url"],
            'overlayurl': report["overlay_url"]         # NEW
        }).execute()
    except Exception as e:
        raise

# ── Convert Report ────────────────────
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

# ── No Crack Report ────────────────────
def noCrackReport(report):
    print(f"No crack report = {report}")
    cracklessReport = {
        "image_id": report["image_id"],
        "severity": "N/A",
        "num_crack_regions": 0,
        "largest_crack_area_ratio": 0,
        "largest_crack_length": 0,
        "damage_level": 0,
        "report_status": "None",            # FIXED: was "reportStatus" (camelCase bug)
        "risk_assessment": "N/A",
        "repair_actions": "N/A",
        "inspection_schedule": "N/A",
        "image_url": report["image_url"],
        "mask_url": report["mask_url"],
        "overlay_url": report["overlay_url"]  # NEW
    }
    return cracklessReport

# ── Change Row Status ────────────────────
def changeRowStatus(supabase, id, status):
    if status != "Pending" and status != "None":
        print("status must be Pending or None")
        raise Exception
    try:
        supabase.table("crack_reports").update({"reportstatus": status}).eq("id", id).execute()
        print(f"Called change row for {id} to {status}")
    except Exception as e:
        print(f"Error updating row {id} to {status}. Error: {e}")

# ── Generate AI Report ────────────────────
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

# ── Update report Columns ────────────────────
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