'''
Created By Angie Hollingworth
6 may 2023

Download and save file locally on your computer.
Run it to create a JSON file required for your usecase. The JSON file will save where you save this script.

Make sure the name you add as the title of your use case is the same as your notebook.

Move the JSON file into your repor folder, and copy with notebook, HTML (optional folders) into the usecasses folder when your ready to publish/republish.

*** DO NOT save this script in your repo!! ***

'''

import json
import pathlib



path = pathlib.Path().resolve()
print(path)

title = ''
name = ''
description = ''
tags = []
difficultly = ''
technology = []
datasets = []

i = input("Enter the Title for you usecase: ")
name = i
title = name.replace(' ', '-').lower()

d = input("\nEnter a short description for your use case (<100 characters): ")
description = d[:100]

diff = ['Beginner', 'Intermediate', 'Advanced']

print("\nSelect 1, 2 or 3 from the following for level of difficulty")
hard = input(f'1: {diff[0]}, 2: {diff[1]}, 3: {diff[2]}:\t')
correct = False
while correct ==False:
    try:
        choice = int(hard)
        if choice in [1,2,3]:
            correct = True
        else:
            hard = input(f'1: {diff[0]}, 2: {diff[1]}, 3: {diff[2]}:\t')
    except:
        hard = input(f'1: {diff[0]}, 2: {diff[1]}, 3: {diff[2]}:\t')

difficultly = diff[choice-1]
print(f'\n{difficultly}\n')


print('\nEnter a tag (one at a time) such as "folium", "maps" or "safety" etc. Hit ENTER after each and ENTER when finished:')
fin = False
while fin == False:
    t = input("tag: ")
    if t =="":
        fin = True
    else:
        tags.extend([t.lower()])
print(tags)

print('\nEnter a technology used (one at a time) such as "geojson", "seaborn", or "pandas" etc. Hit ENTER after each and ENTER when finished:')
fin = False
tech_hold = []
while fin == False:
    t = input("technology: ")
    if t =="":
        fin = True
    else:
        tech_hold.extend([t.lower()])
print(tech_hold)

technology.extend({"name":n,"code":"python"}for n in tech_hold)

print('\nEnter the name for each CoM dataset used (one at a time) such as "pedestrian-traffic". Hit ENTER after each and ENTER when finished:')
fin = False
while fin == False:
    ds = input("dataset: ")
    if ds =="":
        fin = True
    else:
        datasets.extend([ds.lower()])
print(datasets)

#Write to JSON, save in python file location
meta = {'title':title, 'name':name, 'description':description, 'tags':tags, 'difficultly':difficultly, 'technology':technology, 'datasets':datasets}

json_object = json.dumps(meta, indent = 4) 
with open(f"{path}/{title}.json", "w") as outfile:
    outfile.write(json_object)