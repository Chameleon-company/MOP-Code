from collections import deque
from PIL import Image
from skimage.morphology import skeletonize
import numpy as np
import os
import statistics
from pathlib import Path
from severity import Width_Height_Based_Severity
#-------------------Params--------------------
SUPPORTED_IMAGE_FORMATS = ("PNG", "JPG", "JPEG")
#Defines how big a crack has to be in order to be detected
CRACK_DETECTION_THRESHOLD = 200


#SEVERITY CALCULATIONS


severityCalculator = Width_Height_Based_Severity()


def convertMask(_mask):
    if _mask.format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Expected PNG, JPG or JPEG file but got {_mask.format}")
    
    binaryMask = _mask.convert("L")
    
    mask = []
    for row in range(binaryMask.height):
        currentRow = []
        for column in range(binaryMask.width):
            pixel = binaryMask.getpixel((column, row))
            if pixel > 127:
                currentRow.append(1)
            else:
                currentRow.append(0)
            
        mask.append(currentRow)
        
        
    return mask


#TO BE DELETED CLASS
#Currently unused class, kept for now incase we need to store the individual crack masks. 
class crackMask():
    rows: int
    cols: int
    mask: list[list[int]]
    
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.mask = [[0] * cols for _ in range(rows)]


class individualCrackReport():
    crackLength: int
    crackPixelCount: int
    severity: str
    damageLvl: float
    
    def __init__(self, crackLength, crackPixelCount, severity, damageLvl):
        self.crackLength = crackLength
        self.crackPixelCount = crackPixelCount
        self.severity = severity
        self.damageLvl = damageLvl



def locateIndividualCracks(mask):
    #Crack Count is to keep track of all cracks found
    crackCount = 0
    #This tracks the total size of all cracks combined. 
    totalCrackPixelCount = 0
    
    #Masks is an array of all individual masks
    individualCrackMasks = []
    
    #Rows and Cols define the mask size
    #Visited keeps track of all 1 pixels 
    rows = len(mask)
    cols = len(mask[0])
    visited = set()
    #directions = [(-1,0), (1,0), (0,-1), (0,1)]
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,1), (1,-1), (-1,-1)]
    
    
    def bfs(startRow, startColumn):
        global CRACK_DETECTION_THRESHOLD
        nonlocal totalCrackPixelCount
        queue = deque([(startRow, startColumn)])
        visited.add((startRow, startColumn))
        
        #Create a new crack mask class
        crack_mask = [[0] * cols for _ in range(rows)]
        crackPixelCount = 0
        
        while queue:
            row, column = queue.popleft()
            for rowChange, columnChange in directions:
                newRow, newColumn = row + rowChange, column + columnChange
                if (0 <= newRow < rows and 0 <= newColumn < cols and (newRow, newColumn) not in visited):
                    if mask[newRow][newColumn] == 1:
                        #Adds 1 to totalPixel and Individual pixel counts and adds the pixel to the mask
                        crackPixelCount += 1
                        totalCrackPixelCount += 1
                        crack_mask[newRow][newColumn] = 1

                        visited.add((newRow, newColumn))
                        queue.append((newRow, newColumn))
        
        if crackPixelCount > CRACK_DETECTION_THRESHOLD: 
            crackLength = skeleton(crack_mask) 
            
            #Get mean width and height of crack and then convert them to percentage of total width and height of mask
            _crack_mask = np.array(crack_mask)
            rowWidths = _crack_mask.sum(axis=1)
            colHeights = _crack_mask.sum(axis=0)
            meanWidth = (rowWidths[rowWidths > 0] / cols).mean() * 100
            meanHeight = (colHeights[colHeights > 0] / rows).mean() * 100
            
            damageLvl, severity = severityCalculator.calculateSeverity(rows, cols, crackPixelCount, crackLength, meanWidth, meanHeight)       
            crackReport = individualCrackReport(crackLength, crackPixelCount, severity, damageLvl)
            individualCrackMasks.append(crackReport)            
              
    for row in range(rows):
        for column in range(cols):
            if mask[row][column] == 1 and (row, column) not in visited:
                crackCount += 1
                bfs(row, column)
                    
    
    return crackCount, totalCrackPixelCount, individualCrackMasks



def skeleton(mask):
    mask = np.array(mask).astype(bool)
    skeleton = skeletonize(mask)
    
    return getSkeletonLength(skeleton)
    

def getSkeletonLength(skeleton):
    rows = len(skeleton)
    cols = len(skeleton[0])
    pixels = 0
    for row in range(rows):
        for col in range(cols):
            if skeleton[row][col] == 1:
                pixels += 1

    return pixels



def printForTesting(crackMasks, totalCrackPixelCount, mask):
    if len(crackMasks) == 0:
        print("NO CRACKS DETECTED")
        return 
    
    print(f"Crack Count = {len(crackMasks)}")
    print(f"Total Crack Pixels = {totalCrackPixelCount}")
    
    i = 0
    while i < len(crackMasks):
        print(f"Mask {i + 1} pixel count = {crackMasks[i].crackPixelCount}, length = {crackMasks[i].crackLength}, width ratio = {crackMasks[i].crackPixelCount / crackMasks[i].crackLength}")
        #defineCrackSeverity(len(mask), len(mask[0]), crackMasks[i])
        i += 1
        
        
def generateMetricReport(mask, imgName):
    if mask.format not in SUPPORTED_IMAGE_FORMATS and not isinstance(mask, list):
        raise TypeError
     
    if mask.format in SUPPORTED_IMAGE_FORMATS:
        mask = convertMask(mask)
    
    
    #locateIndividualCracks creates a mask of each crack and finds its size (total pixels)
    crackCount, totalCrackPixelCount, crackMasks = locateIndividualCracks(mask)

    largest_damage_level = 0
    largest_crack_area = 0
    largest_length = 0
    severity = ""
    for mask in crackMasks:
        if mask.damageLvl > largest_damage_level:
            largest_damage_level = mask.damageLvl
            largest_crack_area = mask.crackPixelCount / mask.crackLength
            largest_length = mask.crackLength
            severity = mask.severity
            
            
    #print crack metrics
    if False:
        printForTesting(crackMasks, totalCrackPixelCount, mask)
    
    print(f"Image name = {imgName}")
    #Return
    if len(crackMasks) == 0:
        return {
            "image_id": imgName,
            "crack_detected": False,
            "num_crack_regions": 0,
            "largest_crack_area_ratio": 0, 
            "largest_crack_est_length": 0,
            "severity": "Nan",
            "damage_level": 0
        }
    else:
        return {
            "image_id": imgName,
            "crack_detected": True,
            "num_crack_regions": len(crackMasks),
            "largest_crack_area_ratio": round(largest_crack_area, 2), 
            "largest_crack_est_length": round(largest_length, 2),
            "severity": severity,
            "damage_level": round(float(largest_damage_level), 2)
        }
    

if __name__ == '__main__':
    folder = Path("masks")
    
    reports = []
    
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            img = Image.open(img_path)
            img_name = img_path.stem
            
            report = generateMetricReport(img, img_name)
            reports.append(report)
    
    reports = sorted(reports, key=lambda x: x["damage_level"])
    for report in reports:
        print(report)
