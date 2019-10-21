# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:56:50 2019

@author: Jianmu
"""
#%%
import model_VGG19 as vgg19
#import model_VGG16 as vgg16
import load_img

work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
w = 224
h = 224
x_healthy,y_healthy,x_diabete,y_diabete = load_img.get_img(work_dir,w,h)

mode = ["16_dropout_1","32_dropout_16_1","32_dropout_16_dropout_1"]
#epchos = [50,100,150]
epchos = [100,150]
#learning_rate = [0.0001,0.0005,0.001,0.005]
learning_rate = [0.0001,0.001]
optimizer = ["adam","sgd","rmsprop"]
q = 0;
#for n in range(3):
#    for i in range(3):
#        for j in range(4):
#            for k in range(3):
#                q = q+1
#                vgg16.run_model(q,epchos[k],learning_rate[j],optimizer[n],mode[i],
#                                x_healthy,y_healthy,x_diabete,y_diabete)

for n in range(2):
    for i in range(2):
        for j in range(3):
            for k in range(3):
                q = q+1
                vgg19.run_model(q,epchos[i],learning_rate[n],optimizer[j],mode[k],
                                x_healthy,y_healthy,x_diabete,y_diabete)  