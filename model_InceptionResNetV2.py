# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
from PIL import Image, ImageOps
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2

#%%
#get the directory of the current script
work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"

os.chdir(work_dir)
healthy_dir  = work_dir + "/Healthy"
diabete_dir  = work_dir + "/Diabete"


print(work_dir)
#%%
#Assign image parameters
w = 139
h = 139
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
#%%
#set up training and test dataset
np.random.seed(1228)
size_test = 30
n_healthy = x_healthy.shape[0]
n_diabete = x_diabete.shape[0]
index_healthy = np.random.choice(n_healthy,size_test,replace=False)
index_diabete = np.random.choice(n_diabete,size_test,replace=False)

x_test = np.vstack((x_healthy[index_healthy,:,:,:],x_diabete[index_diabete,:,:,:]))
y_test = np.vstack((y_healthy[index_healthy,:],y_diabete[index_diabete,:]))

#%%

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
print(x_train.shape,x_test.shape)

#%%
##### Create transfer learning model
shape = x_train.shape[1:4]

base_model =InceptionResNetV2(weights='imagenet',
                          include_top=False,
                          input_shape= shape)
        

        
top_model = Sequential( )
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(32, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

transfer_model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

n_layer = len(base_model.layers)
for layer in transfer_model.layers[:n_layer - 1 + 1]:
    layer.trainable = False
    
print("Number of trainable layers:",1)
transfer_model.summary()

#%%
n_epochs = 30
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
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy of InceptionResNetV2 Transfer Learning')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Cross_validation'], loc='upper left')

preds = diabete_model.evaluate(x = x_test, y = y_test)
print(round(preds[1],2))
#%%
print(diabete_model.evaluate(x_test,y_test))

   


