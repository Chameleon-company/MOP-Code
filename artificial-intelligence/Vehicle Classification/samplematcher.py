# Sample Matcher
# By Lucas Kocon

# Background:
# The Yolov8 object-detection algorithm we have used for this project, requires two files:
# an image file, and a text file of the same name containing an integer index value from 0 to n-1,
# and 4 floating-point numbers representing a rectangle coordinates normalised to the given image's dimensions;
# it is possible to have multiple entries in a single image and they can have a different index value too.


# Goal of this script:
# Since a sample in Yolov8 needs an image in an 'images' directory, and a paired text file in the 'labels' directory,
# this program performs a simple check if all the image files in either 'train' or val' groups have mathcing pairs.
# If there are extra text or image files, this program will list the extra file from either one
# This program is not sophisticated by way of only showing either problem but not both at the same time,
# and it has a flaw that it won't find unmatched image and label files if the count of both directories are equal.
# This is flawed code that is quickly made to solve a certain problem in the final week of T2 2024,
# hence it isn't robust and can be an extraneous task for the next contributor to create a better program.

from os import listdir
from os.path import isfile, join

path = 'C:\\Users\\lkoco\\Documents\\MOP-Code\\artificial-intelligence\\Vehicle Classification\\'
key = 'light_van'
tv = 'train'
path1 = path+key+'\\'+tv+'\\labels'
path2 = path+key+'\\'+tv+'\\images'


labels = [f for f in listdir(path1) if isfile(join(path1, f))]
images = [f for f in listdir(path2) if isfile(join(path2, f))]

count1 = len(images)
count2 = len(labels)

print(key + " - " + tv)
if count1 == count2:
    print("No extras")
else:
    if count2 < count1:
        print("Extra image files")
        for i in range(count1):
            temp1 = images[i]
            if temp1[-4:-1] == 'web':
                temp1 = temp1[0:-5]
            else:
                temp1 = temp1[0:-4]
            images[i] = temp1
        templist = images
        for j in range(count2):
            temp2 = labels[j]
            temp2 = temp2[0:-4]
            labels[j] = temp2
            for k in range(len(templist)):
                if templist[k] == temp2:
                    del templist[k]
                    break
    elif count1 < count2:
        print("Extra label files")
        for j in range(count2):
            temp2 = labels[j]
            temp2 = temp2[0:-4]
            labels[j] = temp2
        templist = labels
        for i in range(count1):
            temp1 = images[i]
            if temp1[-4:-1] == 'webp':
                temp1 = temp1[0:-5]
            else:
                temp1 = temp1[0:-4]
            images[i] = temp1
            for k in range(len(templist)):
                if templist[k] == temp1:
                    del templist[k]
                    break
    for l in range(len(templist)):
        print(templist[l])
