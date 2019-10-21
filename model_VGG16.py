# -*- coding: utf-8 -*-

import os
#from PIL import Image, ImageOps
import time
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras import optimizers
from keras.applications.vgg16 import VGG16

#%%
def run_model(q,n_iter,l_r,opt,transfer_mode,x_healthy,y_healthy,x_diabete,y_diabete):
#    print("I am 16")
    K.clear_session()
    #set up training and test dataset
    seed_value = 1228+q
    np.random.seed(seed_value)
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
    print(x_train.shape,x_test.shape)
    #%%
    # parameters
    n_epochs = n_iter
    batch_size = 30
    validation_split = 0.2
    dropout = 0.5
    #%%
    
    ##### Create transfer learning model
    shape = x_train.shape[1:4]
    
    base_model =VGG16(weights='imagenet',
                              include_top=False,
                              input_shape=shape)
            
    
    # layers neural
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    if transfer_mode is "16_dropout_1":
        base_model_name = "vgg16_16_dropout_1"
        top_model.add(Dense(16, activation='relu'))
        top_model.add(Dropout(dropout))
    elif transfer_mode is "32_dropout_16_1":
        base_model_name = "vgg16_32_dropout_16_1"
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
    else:
        base_model_name = "vgg16_32_dropout_16_dropout_1"
        top_model.add(Dense(32, activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
        top_model.add(Dropout(dropout))
        
    top_model.add(Dense(1, activation='sigmoid'))
    transfer_model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
    
    n_layer = len(base_model.layers)
    for layer in transfer_model.layers[:n_layer - 1 + 1]:
        layer.trainable = False
        
    print("Number of trainable layers:",1)
    transfer_model.summary()
    
    #%%
    start = time.time()
    # optimizer, learning rate
    diabete_model = transfer_model
    
    if opt=="adam":
        adam = optimizers.Adam(l_r)
        diabete_model.compile(optimizer = adam,momentum=0.9,
                      loss = "binary_crossentropy", metrics = ["accuracy"])
    elif opt=="sgd":
        sgd = optimizers.SGD(l_r)
        diabete_model.compile(optimizer = sgd,
                      loss = "binary_crossentropy", metrics = ["accuracy"])
    else:
        rmsprop = optimizers.RMSprop(l_r)
        diabete_model.compile(optimizer = rmsprop,
                      loss = "binary_crossentropy", metrics = ["accuracy"])
    
    history = diabete_model.fit(x = x_train, y = y_train,  
                                validation_split=validation_split, 
                                epochs = n_epochs, batch_size = batch_size)
        
    end = time.time() - start
    
    print("time consumes:",end)
    #%%
    plot_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification\\vgg16\\"+base_model_name+"\\"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    os.chdir(plot_dir)
    
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    
    preds = diabete_model.evaluate(x = x_test, y = y_test)
    epochs = "n_epochs="+str(n_epochs)
    lrs = "learning_rate="+str(l_r)
    opts = "optimizer="+opt
    test_eval = "accu"+str(round(preds[1],4))
    plt.text(n_epochs-30,0.62,opts)
    plt.text(n_epochs-30,0.6,lrs)
    plt.text(n_epochs-30,0.58,epochs)
    plt.text(n_epochs-30,0.66,test_eval)
    
    plt.title('Model Accuracy of VGG16 in Different Parameters')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Cross_validation'], loc='upper left')
    plt_path = "./opt="+opt+";lr="+str(l_r)+";epoch="+str(n_epochs)+".png"
    plt.savefig(plt_path)
    
#    print(round(preds[1],4))
    history.history={}
#%%
#if __name__ == "__main__":
#    mode = ["16_dropout_1","32_dropout_16_1","32_dropout_16_dropout_1"]
#    epchos = [2,3,4]
#    learning_rate = [0.0001,0.0005,0.001,0.005]
#    optimizer = ["adam","sgd","rmsprop"]
#    q = 0;
#    for n in range(3):
#        for i in range(3):
#            for j in range(4):
#                for k in range(3):
#                    q = q+1
#                    run_model(q,epchos[k],learning_rate[j],optimizer[n],mode[i])          

   

   


