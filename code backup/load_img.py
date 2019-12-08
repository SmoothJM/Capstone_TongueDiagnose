# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:05:45 2019

@author: Jianmu
"""
import os
from PIL import Image, ImageOps
import time
import numpy as np

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
#%%
#get the directory of the current script
def get_img(work_dir,w=221,h=221):
#    work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
    os.chdir(work_dir)
#    healthy_dir  = work_dir + "/Healthy"
#    mild_dir  = work_dir + "/Mild"
#    moderate_dir  = work_dir + "/Moderate"
#    severe_dir  = work_dir + "/Severe"

    
    #read the healthy data
#    m = len(os.listdir(healthy_dir))
#    x_healthy = np.zeros((m,w,h,3))
#    y_healthy = np.zeros((m,1))
    
    
#    start = time.time()
    
#    i = 0
#    print("Reading ",m, " healthy images...")
#    for pic in os.listdir(healthy_dir):
#        img = Image.open(healthy_dir + "/" + pic)
#        img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
#        img = np.array(img)
#        x_healthy[i,:,:,:] = img
#        i = i+1    
#        
#    
#    #read the diabetes data
#    m = len(os.listdir(diabete_dir))
#    x_diabete = np.zeros((m,w,h,3))
#    y_diabete = np.ones((m,1))
#    
#    
#    i = 0
#    print("Reading ",m, " diabete images...")
#    for pic in os.listdir(diabete_dir):
#        img = Image.open(diabete_dir + "/" + pic)
#        img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
#        img = np.array(img)
#        x_diabete[i,:,:,:] = img
#        i = i+1    
#        
#    run_time = time.time() - start
    
#    print("Finish reading images in ",round(run_time)," seconds")
        
    train_dir = work_dir + "/data/train"
    test_dir = work_dir + "/data/test"
    image_size = (h,w)
    
    train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_set = train_datagen.flow_from_directory(train_dir,
                                                  target_size=image_size,
                                                  batch_size=30,
                                                  class_mode="categorical",
                                                  shuffle=True,
                                                  seed=int(time.time()))
    test_set = test_datagen.flow_from_directory(test_dir,
                                                  target_size=image_size,
                                                  batch_size=30,
                                                  class_mode="categorical",
                                                  shuffle=True,
                                                  seed=int(time.time()))
    
    return (train_set,test_set)