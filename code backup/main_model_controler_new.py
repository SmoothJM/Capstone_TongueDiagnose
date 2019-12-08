# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:56:50 2019

@author: Jianmu
"""
#%%
#import model_VGG19 as vgg19
#import model_VGG16 as vgg16
import model_classification as mt
import load_img
import time
work_dir = "\\Users\\14534\\Desktop\\Capstone Project\\Jianmu Deng\\classification"
w = 48
h = 48
#x_healthy,y_healthy,x_diabete,y_diabete = load_img.get_img(work_dir,w,h)
train_set,test_set = load_img.get_img(work_dir,w,h)
#mode = ["16_dropout_1","32_dropout_16_1","32_dropout_16_dropout_1"]
#mode = ["64_dropout_32_16_1",
#        "64_32_dropout_16_1",
#        "64_dropout_32_dropout_16_1",
#        "64_dropout_32_dropout_16_dropout_1"]
mode = ["64_dropout_32_dropout_16_1",
        "64_dropout_32_16_1",
        "16_dropout_1",
        "32_dropout_16_1"]
#mode_ = ["64_32_dropout_16_1",
#         "32_dropout_16_1"]
#epchos = [50,100,150]
#epchos = [100,150]
epchos = [120]
dropout_rate=[0.5,0.7]
#learning_rate = [0.0001,0.0005,0.001,0.005]
learning_rate = [0.00015, 0.0001]
optimizer = ["adam","rmsprop"]
#models=["VGG16","VGG19","DenseNet","InceptionResNetV2",
#        "InceptionV3","NASNet","ResNet50","Xception"]
#models=["VGG16","VGG19","NASNet","Xception"]
#models=["InceptionResNetV2","InceptionV3"]
models=["DenseNet","ResNet50"]
# models_=["NASNet","Xception"]#models_=["NASNet","Xception"]

img_shape = (w,h,3)
start = time.time()
q = 0;
mt.run_model(time.time(),"VGG16",120,0.0001,"adam","32_dropout_16_1",0.5,img_shape,train_set,test_set)
## vgg 16
#for n in range(3):
#    for i in range(2):
#        for j in range(2):
#            for k in range(4):
#                q = q+1
#                vgg16.run_model(q,epchos[0],learning_rate[i],optimizer[n],mode[k],dropout_rate[j],
#                                x_healthy,y_healthy,x_diabete,y_diabete)
## vgg 19
#for n in range(3):
#    for i in range(2):
#        for j in range(2):
#            for k in range(4):
#                q = q+1
#                vgg19.run_model(q,epchos[0],learning_rate[i],optimizer[n],mode[k],dropout_rate[j],
#                                x_healthy,y_healthy,x_diabete,y_diabete)  

# total model test
#for m in range(len(models)):
#    for n in range(len(optimizer)):
#        for i in range(len(learning_rate)):
#            for j in range(len(dropout_rate)):
#                for k in range(len(mode)):
#                    mt.run_model(time.time(),models[m],epchos[0],learning_rate[i],optimizer[n],mode[k],dropout_rate[j],
#                                    x_healthy,y_healthy,x_diabete,y_diabete)
# only for sgd
#for m in range(len(models)):
#    for j in range(len(dropout_rate)):
#        for k in range(len(mode)):
#            mt.run_model(time.time(),models[m],epchos[0],0.001,"sgd",mode[k],dropout_rate[j],
#                            x_healthy,y_healthy,x_diabete,y_diabete)
#for m in range(len(models_)):
#    for n in range(len(optimizer)):
#        for i in range(len(learning_rate)):
#            for j in range(len(dropout_rate)):
#                for k in range(len(mode_)):
#                    mt.run_model(time.time(),models_[m],epchos[0],learning_rate[i],optimizer[n],
#                                 mode_[k],dropout_rate[j],x_healthy,y_healthy,x_diabete,y_diabete)
#for m in range(len(models_)):
#    for j in range(len(dropout_rate)):
#        for k in range(len(mode_)):
#            mt.run_model(time.time(),models_[m],epchos[0],0.001,"sgd",mode_[k],dropout_rate[j],
#                            x_healthy,y_healthy,x_diabete,y_diabete)
#%%
end = time.time() - start
print("time consumes:",end)
file = open("time.txt","w+")
file.write(str(end))
file.close()