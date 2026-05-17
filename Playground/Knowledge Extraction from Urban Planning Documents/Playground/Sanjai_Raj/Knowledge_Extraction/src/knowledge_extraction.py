import re

def extract_knowledge(text, document_name):
    """
    Sanjai's Optimized Model: Uses a 'Keyword-First' approach to ensure
    Policy Areas and Zones are caught even without strict labels.
    """
    
    # QUALITY CHECK: Skip noise fragments
    if len(str(text).split()) < 3:
        return [{
            "document_name": document_name,
            "Zoning": "Not Found",
            "Max_Building_Height": "Not Found",
            "Proposed_Parking": "Not Found",
            "Environmental_Overlay": "Not Found"
        }]

    # 1. Zoning - Redesigned to find keywords anywhere in the sentence
    # This will catch 'Township Policy Area' or 'Urban Growth Boundary' directly
    zoning_keywords = r"(?:Township\s+[A-Z]|Policy\s+Area|Urban\s+Growth\s+Boundary|C1Z|GRZ|NRZ|Residential|Industrial|Rural|Farming|Commercial)"
    zoning_match = re.search(f"({zoning_keywords}(?:\s+Zone)?)", text, re.IGNORECASE)
    zoning = zoning_match.group(1).strip() if zoning_match else "Not Found"

    # Safety Filter: Ensure we didn't grab a fragment of a word
    if zoning.lower() in ['shall', 'must', 'will', 'remai', 'or', 'and', 'the', 'is', 'of']:
        zoning = "Not Found"

    # 2. Height - Broadened for planning constraints
    height_match = re.search(r"(?:height|limit|exceed|maximum|slope).{0,20}(\d+(?:\.\d+)?\s*(?:%|m|metres|meters|storeys|stories))", text, re.IGNORECASE)
    height = height_match.group(1).strip() if height_match else "Not Found"

    # 3. Parking
    parking_match = re.search(r"(\d+)\s*(?:car\s*)?(?:spaces?|parking|bays?|garages?)", text, re.IGNORECASE)
    parking = parking_match.group(1).strip() if parking_match else "Not Found"

    # 4. Environmental Overlays 
    overlay_match = re.search(r"([A-Za-z\s]+Overlay(?:\s+-\s+Schedule\s+\d+)?)", text, re.IGNORECASE)
    overlay = overlay_match.group(1).strip() if overlay_match else "Not Found"

    return [{
        "document_name": document_name,
        "Zoning": zoning,
        "Max_Building_Height": height,
        "Proposed_Parking": parking,
        "Environmental_Overlay": overlay
    }]