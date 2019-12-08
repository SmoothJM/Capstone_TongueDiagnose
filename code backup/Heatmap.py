# -*- coding: utf-8 -*-
"""
Author: Song Chen

This is the main file of the CNN model
"""

#import library
import os
import sys
import numpy as np
from PIL import Image, ImageOps
import time
import matplotlib.pyplot as plt
from keras import applications
from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout
from keras.models import Model
import keras





import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import backend as K

from heatmap import to_heatmap, synset_to_dfs_ids

#%%

#get the directory of the current script

##mac
work_dir = "/Users/schen/Dropbox/Projects/Diabetes/main/classification"
os.chdir(work_dir)
healthy_dir  = work_dir + "/Healthy"
diabete_dir  = work_dir + "/Diabete"

#Windows
#work_dir = "C:\\Users\\Song\\Dropbox\\Projects\\Diabetes\\main\\classification"
#os.chdir(work_dir)
#healthy_dir  = work_dir + "\\Healthy"
#diabete_dir  = work_dir + "\\Diabete"

print(work_dir)
#%%
#Assign image parameters
w = 128
h = 128
image_size = (w,h)


#read the healthy data
m = len(os.listdir(healthy_dir))
x_healthy = np.zeros((m,w,h,3))
y_healthy = np.zeros((len(x_healthy),1))


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
y_diabete = np.ones((len(x_diabete),1))


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

#set up training and test dataset
np.random.seed(1228)
size_test = 30
n_healthy = x_healthy.shape[0]
n_diabete = x_diabete.shape[0]
index_healthy = np.random.choice(n_healthy,size_test,replace=False)
index_diabete = np.random.choice(n_diabete,size_test,replace=False)

x_test = np.vstack((x_healthy[index_healthy,:,:,:],x_diabete[index_diabete,:,:,:]))
y_test = np.vstack((y_healthy[index_healthy,:],y_diabete[index_diabete,:]))


x_train = np.vstack((np.delete(x_healthy,index_healthy,0), np.delete(x_diabete,index_diabete,0)))
y_train = np.vstack((np.delete(y_healthy,index_healthy,0),np.delete(y_diabete,index_diabete,0)))

n_train = x_train.shape[0]
n_test = x_test.shape[0]

#Shuffle by row
shuffle = np.random.permutation(n_train)

x_train = np.take(x_train,shuffle,axis=0,out=x_train)
y_train = np.take(y_train,shuffle,axis=0,out=y_train)

#Normalize the image (divide by 255)
x_train = x_train/255
x_test = x_test/255


print("Number of training examples: ", n_train)
print("Number of test examples: ", n_test)
print(x_train.shape)

#%%
##### Create transfer learning model
shape = x_train.shape[1:4]

base_model = applications.VGG16(weights='imagenet',
                          include_top=False,
                          input_shape= shape)
        

        
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(32, activation='relu'))
top_model.add(Dropout(dropout))
top_model.add(Dense(1, activation='sigmoid'))

transfer_model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

n_layer = len(base_model.layers)
for layer in transfer_model.layers[:n_layer - 1 + 1]:
    layer.trainable = False
    
print("Number of trainable layers:",1)
transfer_model.summary()

#%%
n_epochs = 1
batch_size = 30
validation_split = 0.2
accuracy = []
dropout = 0.5


start = time.time()
    
diabete_model = transfer_model
diabete_model.compile(optimizer = "Adam",
                  loss = "binary_crossentropy", metrics = ["accuracy"])

history = diabete_model.fit(x = x_train, y = y_train,  
                            validation_split=validation_split, 
                            epochs = n_epochs, batch_size = batch_size)
    
end = time.time() - start

#%%
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
#%%
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = healthy_dir+"/070917.jpg"

img = image.load_img(img_path, target_size=(128, 128))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

model = diabete_model
preds = model.predict(x/255)

#%%



african_elephant_output = model.output

last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

#%%

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

#%%
import cv2

#img = cv2.imread(img_path)
img = image.load_img(img_path, target_size=(128, 128))

img = image.img_to_array(img)



heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

#%%

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img
plt.imshow(superimposed_img)

#cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)


#%%
#plt.imshow(img)
plt.imshow(heatmap)







   








