# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 16:05:45 2019

@author: Jianmu
"""
import os
from PIL import Image, ImageOps

import numpy as np
import keras
#%%
#get the directory of the current script
def get_img(work_dir,w=221,h=221):
#    work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
    n_class=4
    image_size = (h,w)
    #%%
    # Intialization all arrays
    # train set path
    os.chdir(work_dir)
    healthy_train_dir  = work_dir + "\\data\\train\\Healthy"
    mild_train_dir  = work_dir + "\\data\\train\\Mild"
    moderate_train_dir  = work_dir + "\\data\\train\\Moderate"
    severe_train_dir  = work_dir + "\\data\\train\\Severe"
    # test set path
    healthy_test_dir  = work_dir + "\\data\\test\\Healthy"
    mild_test_dir  = work_dir + "\\data\\test\\Mild"
    moderate_test_dir  = work_dir + "\\data\\test\\Moderate"
    severe_test_dir  = work_dir + "\\data\\test\\Severe" 
    
    
    
    # train set number
    n_train_healthy = len(os.listdir(healthy_train_dir))
    n_train_mild = len(os.listdir(mild_train_dir))
    n_train_moderate = len(os.listdir(moderate_train_dir))
    n_train_severe = len(os.listdir(severe_train_dir))
    # test set number
    n_test_healthy = len(os.listdir(healthy_test_dir))
    n_test_mild = len(os.listdir(mild_test_dir))
    n_test_moderate = len(os.listdir(moderate_test_dir))
    n_test_severe = len(os.listdir(severe_test_dir))
    
    # train set matrix
    x_train_healthy = np.zeros((n_train_healthy,w,h,3))
    x_train_mild = np.zeros((n_train_mild,w,h,3))
    x_train_moderate = np.zeros((n_train_moderate,w,h,3))
    x_train_severe = np.zeros((n_train_severe,w,h,3))
    
    y_train_healthy = np.zeros((n_train_healthy,1))
    y_train_mild = np.ones((n_train_mild,1))
    y_train_moderate = np.full((n_train_moderate,1), 2.0)
    y_train_severe = np.full((n_train_severe,1), 3.0)
    
    # test set matrix
    x_test_healthy = np.zeros((n_test_healthy,w,h,3))
    x_test_mild = np.zeros((n_test_mild,w,h,3))
    x_test_moderate = np.zeros((n_test_moderate,w,h,3))
    x_test_severe = np.zeros((n_test_severe,w,h,3))
    
    y_test_healthy = np.zeros((n_test_healthy,1))
    y_test_mild = np.ones((n_test_mild,1))
    y_test_moderate = np.full((n_test_moderate,1), 2.0)
    y_test_severe = np.full((n_test_severe,1), 3.0)
    
    # train set one hot
    y_train_healthy = keras.utils.to_categorical(y_train_healthy, n_class)
    y_train_mild = keras.utils.to_categorical(y_train_mild, n_class)
    y_train_moderate = keras.utils.to_categorical(y_train_moderate, n_class)
    y_train_severe = keras.utils.to_categorical(y_train_severe, n_class)
    # test set one hot
    y_test_healthy = keras.utils.to_categorical(y_test_healthy, n_class)
    y_test_mild = keras.utils.to_categorical(y_test_mild, n_class)
    y_test_moderate = keras.utils.to_categorical(y_test_moderate, n_class)
    y_test_severe = keras.utils.to_categorical(y_test_severe, n_class)
    #%%
    # Load image (Fill all arrays)
    
    # load train set
    i=0
    for pic in os.listdir(healthy_train_dir):
            img = Image.open(healthy_train_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_train_healthy[i,:,:,:] = img
            i = i+1 
    i=0
    for pic in os.listdir(mild_train_dir):
            img = Image.open(mild_train_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_train_mild[i,:,:,:] = img
            i = i+1 
    i=0
    for pic in os.listdir(moderate_train_dir):
            img = Image.open(moderate_train_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_train_moderate[i,:,:,:] = img
            i = i+1 
    i=0
    for pic in os.listdir(severe_train_dir):
            img = Image.open(severe_train_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_train_severe[i,:,:,:] = img
            i = i+1 
            
    # load test set
    i=0
    for pic in os.listdir(healthy_test_dir):
            img = Image.open(healthy_test_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_test_healthy[i,:,:,:] = img
            i = i+1 
    i=0
    for pic in os.listdir(mild_test_dir):
            img = Image.open(mild_test_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_test_mild[i,:,:,:] = img
            i = i+1 
    i=0
    for pic in os.listdir(moderate_test_dir):
            img = Image.open(moderate_test_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_test_moderate[i,:,:,:] = img
            i = i+1 
    i=0
    for pic in os.listdir(severe_test_dir):
            img = Image.open(severe_test_dir + "\\" + pic)
            img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
            img = np.array(img)
            x_test_severe[i,:,:,:] = img
            i = i+1 
            
    x_train = np.vstack((x_train_healthy[:,:,:,:],x_train_mild[:,:,:,:],x_train_moderate[:,:,:,:],x_train_severe[:,:,:,:]))
    y_train = np.vstack((y_train_healthy[:,:],y_train_mild[:,:],y_train_moderate[:,:],y_train_severe[:,:]))
    
    x_test = np.vstack((x_test_healthy[:,:,:,:],x_test_mild[:,:,:,:],x_test_moderate[:,:,:,:],x_test_severe[:,:,:,:]))
    y_test = np.vstack((y_test_healthy[:,:],y_test_mild[:,:],y_test_moderate[:,:],y_test_severe[:,:]))

    return (x_train,y_train,x_test,y_test)