# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:14:14 2019

@author: Jianmu
"""

from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
#%%
model = VGG19()
print(model.summary())
#%%
#C:/Users/14534/Desktop/1.jpg
img_path = "C:/Users/14534/Desktop/6.jpg"
image = load_img(img_path, target_size=(224,224))
image_data = img_to_array(image)
image_data = np.expand_dims(image_data, axis=0)
print(image_data.shape)
image_data = preprocess_input(image_data)
#%%
prediction = model.predict(image_data)
results = decode_predictions(prediction,top=5)
print(results)

