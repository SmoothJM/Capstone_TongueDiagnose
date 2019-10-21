# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:02:38 2019

@author: Jianmu
"""
#import keras
import os
#from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img,img_to_array
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model,Sequential
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import applications

from PIL import Image, ImageOps
#%%
work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
os.chdir(work_dir)
healthy_dir  = work_dir + "/Healthy"
diabete_dir  = work_dir + "/Diabete"
#%%
w = 48
h = 48
image_size = (w,h)
#%%
# load picture of healthy and diabetes tongue pictures
# load healthy pictures
m = len(os.listdir(healthy_dir))
x_healthy = np.zeros((m,w,h,3))
y_healthy = np.zeros((m,1))
i=0
print("Loading healthy pictures...")
for pic in os.listdir(healthy_dir):
    img = Image.open(healthy_dir + "/" + pic)
    img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
    img_data = np.array(img)
#    img = load_img(healthy_dir + "/" + pic, target_size=image_size)
#    img_data = img_to_array(img)
    
    x_healthy[i,:,:,:] = img_data
    i += 1
    


# load diabetes pictures
m = len(os.listdir(diabete_dir))
x_diabete = np.zeros((m,w,h,3))
y_diabete = np.zeros((m,1))
i=0
print("Loading diabetes pictures...")
for pic in os.listdir(diabete_dir):
    img = Image.open(diabete_dir + "/" + pic)
    img = ImageOps.fit(img, image_size, Image.ANTIALIAS)
    img_data = np.array(img)
#    img = load_img(diabete_dir + "/" + pic, target_size=image_size)
#    img_data = img_to_array(img)
    x_diabete[i,:,:,:] = img_data
    i += 1
print("Pictures loading is done!")
#%%
#set up training and test dataset
np.random.seed(1228)
size_test = 30
n_healthy = x_healthy.shape[0]
n_diabete = x_diabete.shape[0]
# 不放回的选出30个健康和不健康的舌头索引
index_healthy = np.random.choice(n_healthy,size_test,replace=False)
index_diabete = np.random.choice(n_diabete,size_test,replace=False)

# 此时vstack中的参数是一个由两个长度为三十的数组组成的元祖
# x_test.shape = (60,32,32,3)
x_test = np.vstack((x_healthy[index_healthy,:,:,:],x_diabete[index_diabete,:,:,:]))
y_test = np.vstack((y_healthy[index_healthy,:],y_diabete[index_diabete,:]))
# train中留下了未被选中的
x_train = np.vstack((np.delete(x_healthy,index_healthy,0), np.delete(x_diabete,index_diabete,0)))
y_train = np.vstack((np.delete(y_healthy,index_healthy,0),np.delete(y_diabete,index_diabete,0)))
print("x_train shape is:",x_train.shape,"; x_test shape is:",x_test.shape)
# n_train is 386 = 223*2-60*2
# n_test is 60
n_train = x_train.shape[0]
n_test = x_test.shape[0]

print("Number of training picture is:",n_train)
print("Number of testing picture is:",n_test)
#%%
#Shuffle by row
shuffle = np.random.permutation(n_train)

x_train = np.take(x_train,shuffle,axis=0,out=x_train)
y_train = np.take(y_train,shuffle,axis=0,out=y_train)

#Normalize the image (divide by 255)
x_train = x_train/255
x_test = x_test/255
#%%
##### Create transfer learning model
# shape of each picture; should be 224*224*3
shape = x_train.shape[1:4]
dropout=0.5
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

# 冻结vgg原层参数
for layer in transfer_model.layers[:n_layer]:
    layer.trainable = False
    
print("Number of trainable layers:",len(top_model.layers))
transfer_model.summary()
#%%
# training start
n_epochs = 50
batch_size = 30
validation_split = 0.2
accuracy = []
# rmsprop adam sgd 
diabete_model = transfer_model

diabete_model.compile(optimizer = "adam",
                      loss = "binary_crossentropy", metrics = ["accuracy"])
history = diabete_model.fit(x = x_train, y = y_train,  
                            validation_split=validation_split,
                            epochs = n_epochs, batch_size = batch_size)

score = diabete_model.evaluate(x_test,y_test)
print("Final loss: ",score[0],"; Final accuracy:",score[1])
#%%
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Cross_validation'], loc='upper left')
#    fig.savefig(model_name+".png")

preds = diabete_model.evaluate(x = x_test, y = y_test)
print(round(preds[1],2))
#%%















