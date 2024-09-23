# Index Changer
# By Lucas Kocon

# Background:
# In the beginning of our project work, we initially thought we needed a 30-odd category vehicle classification algorithm,
# but we were mistaken and we needed a 13 category system instead.
# Later on after experimenting with how to train a model,
# I understood how we need to reorganise the files so they can be all placed together,
# which involved a major renaming effort in the final weeks,
# plus some files needed a change in their index value.
# I had to do so when I had a labelling issue in my first Three-Axle Articulated submission,
# where a find-and-replace solution I made with Notepad++ worked ONLY IF the number of changes matched the number of files
# but a robust solution is needed for Jing's Six Axle Articulated label files.

# Goal for this script:
# This script will edit and change the index value of every label file when directed to the path that contains them
# Handy for changing many files in bulk without issues of incidental change in coordinates

from os import listdir
from os.path import isfile, join

path = 'C:\\Users\\lkoco\\Documents\\MOP-Code\\artificial-intelligence\\Vehicle Classification\\'
key = 'vehicle_class_1'
tv = 'train'
pathL = path+key+'\\'+tv+'\\labels\\'
indexResult = '0'

print(pathL)

labels = [f for f in listdir(pathL) if isfile(join(pathL, f))]
print(labels)

for i in range(len(labels)):
    temp = open(join(pathL,labels[i]),"r")
    tempContent = str(temp.read())
    temp.close()
    ##print(tempContent)
    tempFirst = tempContent[0]
    tempSecond = tempContent[1]
    if len(indexResult) == 1:
        if tempFirst != indexResult:
            if tempSecond != ' ':
                print(tempContent)
            tempC = tempContent.replace(tempFirst,indexResult,1)
            tempC = tempC.replace('\n'+tempFirst,'\n'+indexResult)
        print(str(i) +":\n"+str(tempC))
    temp = open(join(pathL,labels[i]),"w")
    temp.write(tempC)
    temp.close()
