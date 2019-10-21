# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:42:46 2019

@author: Jianmu
"""

from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
#%%
model = Xception()
print(model.summary())
#%%
target_size=(299,299)
img_path = "C:/Users/14534/Desktop/1.jpg"
image = load_img(img_path, target_size=target_size)
image_data = img_to_array(image)
#image_data = image_data.reshape((1,) + image_data.shape)
image_data = np.expand_dims(image_data, axis=0)
print(image_data.shape)
image_data = preprocess_input(image_data)
#%%
prediction = model.predict(image_data)
results = decode_predictions(prediction,top=3)
print(results)




  





