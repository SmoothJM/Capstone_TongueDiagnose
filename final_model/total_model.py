# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:25:51 2020

@author: Jianmu
"""
#%%
#import os
#import sys
#current_dir = os.path.abspath(os.path.dirname(__file__))
#sys.path.append(current_dir)
#sys.path.append("..")
from keras.models import load_model
from PIL import Image, ImageOps
#import yolo
import od_predict as od
import model_diagnose as md
import sys

def run_model(image_name):
    od.object_detection(image_name)
    print(md.diagnose(image_name)+" "+image_name)
run_model(sys.argv[1])