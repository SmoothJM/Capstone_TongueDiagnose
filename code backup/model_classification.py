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
from keras.applications.vgg19 import VGG19
#from keras.applications.resnet50 import ResNet50
#from keras.applications.xception import Xception
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.densenet import DenseNet121
from keras import regularizers

#%%
def run_model(q,model_name,n_iter,l_r,opt,transfer_mode,dropout_rate,img_shape,train_set,test_set):
#    print("I am 16")
    l2 = 0.01
    K.clear_session()
    #set up training and test dataset
    seed_value = int(q)
    np.random.seed(seed_value)
#    size_test = 30
#    n_healthy = x_healthy.shape[0]
#    n_diabete = x_diabete.shape[0]
#    index_healthy = np.random.choice(n_healthy,size_test,replace=False)
#    index_diabete = np.random.choice(n_diabete,size_test,replace=False)
#    
#    x_test = np.vstack((x_healthy[index_healthy,:,:,:],x_diabete[index_diabete,:,:,:]))
#    y_test = np.vstack((y_healthy[index_healthy,:],y_diabete[index_diabete,:]))
#    
#    
#    x_train = np.vstack((np.delete(x_healthy,index_healthy,0), np.delete(x_diabete,index_diabete,0)))
#    y_train = np.vstack((np.delete(y_healthy,index_healthy,0),np.delete(y_diabete,index_diabete,0)))
#    
#    n_train = x_train.shape[0]
#    n_test = x_test.shape[0]
#    
#    #Shuffle by row
#    shuffle = np.random.permutation(n_train)
#    
#    x_train = np.take(x_train,shuffle,axis=0,out=x_train)
#    y_train = np.take(y_train,shuffle,axis=0,out=y_train)
#    
#    #Normalize the image (divide by 255)
#    x_train = x_train/255
#    x_test = x_test/255
#    
#    
#    print("Number of training examples: ", n_train)
#    print("Number of test examples: ", n_test)
#    print(x_train.shape,x_test.shape)
    #%%
    # parameters
    n_epochs = n_iter
#    batch_size = 30
#    validation_split = 0.2
    dropout = dropout_rate
    #%%
    
    ##### Create transfer learning model
    shape = img_shape
#    print(model_name)
    if model_name is "VGG16":
        base_model =VGG16(weights='imagenet',
                                  include_top=False,
                                  input_shape=shape)
    elif model_name is "VGG19":
        base_model =VGG19(weights='imagenet',
                                  include_top=False,
                                  input_shape=shape)
    
    
    # layers neural
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    base_model_name = model_name +"_"+ transfer_mode
    if transfer_mode is "16_dropout_1":
#        base_model_name = "vgg16_16_dropout_1"
        top_model.add(Dense(16,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
    elif transfer_mode is "32_dropout_16_1":
#        base_model_name = "vgg16_32_dropout_16_1"
        top_model.add(Dense(32,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
    elif transfer_mode is "32_dropout_16_dropout_1":
#        base_model_name = "vgg16_32_dropout_16_dropout_1"
        top_model.add(Dense(32,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
        top_model.add(Dropout(dropout))
    elif transfer_mode is "64_dropout_32_16_1":
#        base_model_name = "vgg16_64_dropout_32_16_1"
        top_model.add(Dense(64,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(32,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dense(16, activation='relu'))
    elif transfer_mode is "64_32_dropout_16_1":
#        base_model_name = "vgg16_64_32_dropout_16_1"
        top_model.add(Dense(64,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dense(32,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
    elif transfer_mode is "64_dropout_32_dropout_16_1":
#        base_model_name = "vgg16_64_dropout_32_dropout_16_1"
        top_model.add(Dense(64,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(32,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
    elif transfer_mode is "64_dropout_32_dropout_16_dropout_1":
#        base_model_name = "vgg16_64_dropout_32_dropout_16_dropout_1"
        top_model.add(Dense(64,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(32,kernel_regularizer=regularizers.l2(l2), activation='relu'))
        top_model.add(Dropout(dropout))
        top_model.add(Dense(16, activation='relu'))
        top_model.add(Dropout(dropout))
    top_model.add(Dense(4, activation='softmax'))
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
        # is_categorical_crossentropy
        # binary_crossentropy
        diabete_model.compile(optimizer = adam,
                      loss = "categorical_crossentropy", metrics = ["accuracy"])
    elif opt=="sgd":
        sgd = optimizers.SGD(l_r,momentum=0.9)
        diabete_model.compile(optimizer = sgd,
                      loss = "categorical_crossentropy", metrics = ["accuracy"])
    else:
        rmsprop = optimizers.RMSprop(l_r)
        diabete_model.compile(optimizer = rmsprop,
                      loss = "categorical_crossentropy", metrics = ["accuracy"])
    
#    history = diabete_model.fit(x = x_train, y = y_train,  
#                                validation_split=validation_split, 
#                                epochs = n_epochs, batch_size = batch_size)
    history = diabete_model.fit_generator(train_set,
                                          epochs=n_epochs,
                                          validation_data=test_set)
        
    end = time.time() - start
    
    print("time consumes:",end)
    #%%
    model_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification\\"+model_name+"\\"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    plot_dir = model_dir+base_model_name+"\\"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    os.chdir(plot_dir)
    
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    
    preds = diabete_model.evaluate(test_set)
#    epochs = "n_epochs="+str(n_epochs)+";"
    dp ="dropout= "+str(dropout_rate)
    lrs = "learning_rate= "+str(l_r)+"; "
    opts = "optimizer= "+opt+"; "
    test_eval = "accu= "+str(round(preds[1],4))
    plt.text(n_epochs-40,0.63,opts)
    plt.text(n_epochs-40,0.6,lrs)
    plt.text(n_epochs-40,0.57,dp)
    plt.text(n_epochs-40,0.67,test_eval)
    
    plt.title('Model Accuracy of '+model_name +' in Different Parameters')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Cross_validation'], loc='upper left')
    plt_path = "./"+opts+lrs+dp+".png"
    
    plt.savefig(plt_path)
    plt.close()
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

   

   


