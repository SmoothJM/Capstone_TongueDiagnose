# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:00:14 2020

@author: Jianmu
"""

#%%
import total_model as tm
from keras.models import load_model
from PIL import Image, ImageOps
import yolo
import od_predict as od
import model_diagnose as md

image_name = "1.jpg"
result = tm.run_model(image_name)
print(result)