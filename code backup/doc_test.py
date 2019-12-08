# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:31:07 2019

@author: Jianmu
"""

#%%

import pandas as pd
import os
import shutil
import numpy as np
import time
def copy_new_img():
    #get the directory of the current script
    work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
    os.chdir(work_dir)
    df = pd.read_excel("data.xlsx",sheet_name=0,usecols=[0,4])
    #data=DataFrame.from_dict(df,orient='index',columns=['state'])
    
    #Select the healty patient ID according to the status column
    id_healthy = df["index"][df["state"] == "Normal"]
    id_mild = df["index"][df["state"] == "Mild"]
    id_moderate = df["index"][df["state"] == "Moderate"]
    id_severe = df["index"][df["state"] == "Severe"]
    
    #check if it contains all data
    print("Number of healthy patitents,", len(id_healthy))
    print("Number of mild diabete patitents,", len(id_mild))
    print("Number of moderate diabete patitents,", len(id_moderate))
    print("Number of severe diabete patitents,", len(id_severe))
    
    
    source_dir = work_dir + "\\pic"
    healthy_dir  = work_dir + "\\Healthy"
    mild_dir  = work_dir + "\\Mild"
    moderate_dir  = work_dir + "\\Moderate"
    severe_dir  = work_dir + "\\Severe"
    
    shutil.rmtree(healthy_dir,True)
    shutil.rmtree(mild_dir,True)
    shutil.rmtree(moderate_dir,True)
    shutil.rmtree(severe_dir,True)
    
    #create folders
    if os.path.isdir(healthy_dir):
        print("Healthy folder already exists")
    else:
        os.makedirs(healthy_dir)
        print("Created healthy folder")
        
    if os.path.isdir(mild_dir):
        print("Mild diabetes folder already exists")
    else:
        os.makedirs(mild_dir)
        print("Created mild diabetes folder")
    
    if os.path.isdir(moderate_dir):
        print("Moderate diabetes folder already exists")
    else:
        os.makedirs(moderate_dir)
        print("Created moderate diabetes folder")
        
    if os.path.isdir(severe_dir):
        print("Severe diabetes folder already exists")
    else:
        os.makedirs(severe_dir)
        print("Created severe diabetes folder")
    
    #copy the healthy pictures 
    for pat in id_healthy:
        pat = "0"+str(pat)
        file=source_dir+"\\"+pat+".jpg"
        if os.path.exists(file):
            shutil.copy(file,healthy_dir)
        else:
            print("Healthy:",pat,"is not existent")
    
    #copy the diabetes pictures
    for pat in id_mild:
        pat = "0"+str(pat)
        file=source_dir+"\\"+pat+".jpg"
        if os.path.exists(file):
            shutil.copy(file,mild_dir)
        elif os.path.exists(source_dir+"\\"+pat+"a.jpg"):
            shutil.copy(source_dir+"\\"+pat+"a.jpg",mild_dir)
        else:
            print("Mild:",pat,"is not existent")
    for pat in id_moderate:
        pat = "0"+str(pat)
        file=source_dir+"\\"+pat+".jpg"
        if os.path.exists(file):
            shutil.copy(file,moderate_dir)
        elif os.path.exists(source_dir+"\\"+pat+"a.jpg"):
            shutil.copy(source_dir+"\\"+pat+"a.jpg",moderate_dir)
        else:
            print("Moderate:",pat,"is not existent")
    for pat in id_severe:
        pat = "0"+str(pat)
        file=source_dir+"\\"+pat+".jpg"
        if os.path.exists(file):
            shutil.copy(file,severe_dir)
        elif os.path.exists(source_dir+"\\"+pat+"a.jpg"):
            shutil.copy(source_dir+"\\"+pat+"a.jpg",severe_dir)
        else:
            print("Severe:",pat,"is not existent")
    
    number_files_healthy = len(os.listdir(healthy_dir)) 
    number_files_diabete = len(os.listdir(mild_dir)) + len(os.listdir(moderate_dir)) + len(os.listdir(severe_dir))
    n_diff = number_files_diabete - number_files_healthy
    print(n_diff)
    match_dir = work_dir + "\\healthy_tongue"
    match_list = os.listdir(match_dir)
    
    np.random.seed(int(time.time()))
    random_list = np.random.choice(match_list,size = n_diff,replace = False)
    match_path = [os.path.join(match_dir,image) for image in random_list]
    
    for file in match_path:
        shutil.copy(file,healthy_dir)
#%%
copy_new_img()
