# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:05:45 2019

@author: Jianmu
"""
import os
from PIL import Image, ImageOps
import time
import numpy as np
#%%
#get the directory of the current script
def get_img(work_dir,w,h):
#    work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
    os.chdir(work_dir)
    healthy_dir  = work_dir + "/Healthy"
    diabete_dir  = work_dir + "/Diabete"
    print(work_dir)
    #Assign image parameters
    # resolution biger
#    w = 224
#    h = 224
    image_size = (w,h)
    
    #read the healthy data
    m = len(os.listdir(healthy_dir))
    x_healthy = np.zeros((m,w,h,3))
    y_healthy = np.zeros((m,1))
    
    
    start = time.time()
    
    i = 0
    print("Reading ",m, " healthy images...")
    for pic in os.listdir(healthy_dir):
        img = Image.open(healthy_dir + "/" + pic)
        img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
        img = np.array(img)
        x_healthy[i,:,:,:] = img
        i = i+1    
        
    
    #read the diabetes data
    m = len(os.listdir(diabete_dir))
    x_diabete = np.zeros((m,w,h,3))
    y_diabete = np.ones((m,1))
    
    
    i = 0
    print("Reading ",m, " diabete images...")
    for pic in os.listdir(diabete_dir):
        img = Image.open(diabete_dir + "/" + pic)
        img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
        img = np.array(img)
        x_diabete[i,:,:,:] = img
        i = i+1    
        
    run_time = time.time() - start
    
    print("Finish reading images in ",round(run_time)," seconds")
    return (x_healthy,y_healthy,x_diabete,y_diabete)