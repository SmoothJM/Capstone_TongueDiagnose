# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:30:57 2020

@author: Jianmu
"""
#%%
import os
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import sys

def diagnose(img_name):
    h=128
    w=80
    data = np.zeros((1,w,h,3))
    work_dir = "E:/keras-yolo3-master/keras-yolo3-master/final_model"
    os.chdir(work_dir)
    model_path = work_dir + "/weights/vgg.h5"
    img_path = work_dir + "/result/" + img_name
    model = load_model(model_path)
    # model.summary()
    img_size = (h,w)
    img = Image.open(img_path)
    img = ImageOps.fit(img, img_size, Image.ANTIALIAS)
    img = np.array(img)
    data[0,:,:,:] = img
#    print("prediction result: ")
    result = list(model.predict(data)[0])
#    print(result)
    label_index = result.index(max(result))
    label = ["Healthy","Mild","Moderate","Severe"]
    print(label[label_index])
#    return label[label_index]
diagnose(sys.argv[1])
    
    
    