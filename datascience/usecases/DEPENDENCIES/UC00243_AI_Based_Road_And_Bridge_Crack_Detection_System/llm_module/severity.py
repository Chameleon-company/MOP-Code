def classify_severity(length, width):
    if width > 3 or length > 1.5:
        return "High"
    elif width > 1:
        return "Medium"
    return "Low"