# -*- coding: utf-8 -*-
"""
Author: Song Chen

This file reads the image in the "pic" foleder and rewrite them into healthy and diabetes
folder, using the index provided in "data.xlsx"
"""

#import library
import pandas as pd
import os
import sys
import shutil
import numpy as np



#get the directory of the current script
work_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(work_dir)
df = pd.read_excel("data.xlsx", sheet_name=None, skiprows=2)

print("Data file contains ", df.shape[0], " patients")

df = df.dropna(subset = ["状态"])

print("After dropping null status, there are", df.shape[0], " patients left")

#Select the healty patient ID according to the status column
id_healthy = df["编号"][df["状态"] == "正常"]
id_diabete = df["编号"][df["状态"] != "正常"]

#check if it contains all data
print("Number of healthy patitents,", len(id_healthy))
print("Number of diabete patitents,", len(id_diabete))




##set image directory
source_dir = work_dir + "/pic"
healthy_dir  = work_dir + "/Healthy"
diabete_dir  = work_dir + "/Diabete"

#create folders
if os.path.isdir(healthy_dir):
    print("Healthy folder already exists")
else:
    os.makedirs(healthy_dir)
    print("Created healthy folder")
if os.path.isdir(diabete_dir):
    print("Diabetes folder already exists")
else:
    os.makedirs(diabete_dir)
    print("Created diabetes folder")


image_list = os.listdir(source_dir)


#copy the healthy pictures 
for pat in id_healthy:
    pat = str(pat)
    copy_list = [image for image in image_list if pat in os.fsdecode(image)]
    copy_path = [os.path.join(source_dir,image) for image in copy_list]
    for file in copy_path:
        shutil.copy(file,healthy_dir)
        

#copy the diabetes pictures
for pat in id_diabete:
    pat = str(pat)
    copy_list = [image for image in image_list if pat in os.fsdecode(image)]
    copy_path = [os.path.join(source_dir,image) for image in copy_list]
    for file in copy_path:
        shutil.copy(file,diabete_dir)
        
        
number_files_source = len(os.listdir(source_dir)) 
number_files_healthy = len(os.listdir(healthy_dir)) 
number_files_diabete = len(os.listdir(diabete_dir)) 


#count the files
print("Number of pictures in source folder: ", number_files_source)
print("Number of pictures in healthy folder", number_files_healthy)
print("Number of pictures in diabete folder", number_files_diabete)



#match the differences between the healthy and diabetes pictures by randomly sampling 
#form the healthy tongue folder

match_dir = work_dir + "/healthy_tongue"
match_list = os.listdir(match_dir)

number_files_healthy = len(os.listdir(healthy_dir)) 
number_files_diabete = len(os.listdir(diabete_dir)) 

n_diff = number_files_diabete - number_files_healthy
n_pic = len(match_list)

np.random.seed(1228)
random_list = np.random.choice(match_list,size = n_diff,replace = False)
match_path = [os.path.join(match_dir,image) for image in random_list]

for file in match_path:
    shutil.copy(file,healthy_dir)
