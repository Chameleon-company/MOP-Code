# Edge Case Testing — Crack Detection Model

## Overview
The following edge cases were tested by uploading images through the full pipeline — from the UI through the crack detection model, Supabase storage, crack metrics generation, and LLM RAG report generation. All three images (original, binary mask, and crack overlay) were stored in Supabase Storage and the results were saved to the database.

---

## Test Cases

### 1. Normal Crack (Pavement)
**Description:** Standard pavement image with a visible crack running across the surface.  
**Result:** ✅ Passed — clean binary mask and red overlay produced.

![Test 1](test-case-images/test_case_1.jpeg)

---

### 2. Real Life 3D Background
**Description:** Road crack image with a natural background including grass.  
**Result:** ✅ Passed — model handled the real-world environment well with no false positives from the background.

![Test 2](test-case-images/test_case_2.jpeg)

---

### 3. Road Markings (No Crack)
**Description:** Road surface with a white road marking/line but no actual crack.  
**Result:** ✅ Passed — empty mask returned, road marking correctly ignored.

![Test 3](test-case-images/test_case_3.jpeg)

---

### 4. Clean Road (No Crack)
**Description:** Plain road surface with no cracks or markings.  
**Result:** ✅ Passed — no crack detected, empty mask and clean overlay returned.

![Test 4](test-case-images/test_case_4.jpeg)

---

### 5. Pothole
**Description:** Road surface with a pothole rather than a crack.  
**Result:** ✅ Passed — model detected and highlighted the pothole, suggesting it generalises to other types of road damage beyond cracks due to similar dark irregular pixel patterns.

![Test 5](test-case-images/test_case_5.jpeg)

---

## Summary

| Test Case | Result |
|-----------|--------|
| Normal Crack | ✅ Passed |
| Real Life 3D Background | ✅ Passed |
| Road Markings | ✅ Passed |
| Clean Road | ✅ Passed |
| Pothole | ✅ Passed |

The model performed well across all 5 edge cases with no false positives or missed detections. The most notable finding was the model's ability to generalise to potholes, which were not part of the training data, showing the model's robustness to different types of road damage.
