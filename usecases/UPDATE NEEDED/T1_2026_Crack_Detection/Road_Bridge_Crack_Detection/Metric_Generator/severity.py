from abc import ABC, abstractmethod
import math


areaRatioModifier = 0.6
widthScoreModifier = 0.3
lengthScoreModifier = 0.1


SUPERFICIAL = 0
MINOR = 6
MEDIUM = 14
HIGH = 20
SEVERE = 30
CATASTROPHIC = 55


class severityCalc(ABC):
    @abstractmethod
    def calculateSeverity(self, rows, cols, crackPixelCount, crackLength, mean_width, mean_height):
        return float
    

class Width_Height_Based_Severity_No_Height(severityCalc):
    def calculateSeverity(self, rows, cols, crackPixelCount, crackLength, mean_width, mean_height):
        severity = (mean_height + mean_width) / 2 
        return severity
    
class Width_Height_Based_Severity(severityCalc):
    def calculateSeverity(self, rows, cols, crackPixelCount, crackLength, mean_width, mean_height):
        areaRatio = (crackPixelCount / (rows * cols)) * 100
        
        widthScore = mean_width
        relativeImgSize = ((rows**2 + cols**2) ** 0.5)
        lengthScore = math.sqrt(crackLength / relativeImgSize * 100)
        #print(f"Length Score = {lengthScore}, crack length = {crackLength}")
        damageLvl = (areaRatio * areaRatioModifier) + (widthScore * widthScoreModifier) + (lengthScore * lengthScoreModifier)
        if damageLvl < MINOR:
            severity = "Superficial"
        if damageLvl > MINOR and damageLvl < MEDIUM:
            severity = "Minor"
        if damageLvl > MEDIUM and damageLvl < HIGH:
            severity = "Medium"
        if damageLvl > HIGH and damageLvl < SEVERE:
            severity = "High"
        if damageLvl > SEVERE and damageLvl < CATASTROPHIC:
            severity = "Severe"
        if damageLvl > CATASTROPHIC:
            severity = "Catastrophic"
            
        
        
        return damageLvl, severity