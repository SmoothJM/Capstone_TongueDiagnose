from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
#%%
model = VGG16()
print(model.summary())
#%%
#C:/Users/14534/Desktop/1.jpg
target_size=(224,224)
img_path = "C:/Users/14534/Desktop/photo.jpg"
#加载图片，格式变成224*224
image = load_img(img_path, target_size=target_size)

#把图片变成数组
image_data = img_to_array(image)

#image_data = image_data.reshape((1,) + image_data.shape)
#上下两句意思一样，为输入值创建batch维度为1的值
image_data = np.expand_dims(image_data, axis=0)
print(image_data.shape)
#预处理图片，将其转化为VGG-16能够接受的输入，实际上为每个像素减去均值
image_data = preprocess_input(image_data)
#%%
prediction = model.predict(image_data)
results = decode_predictions(prediction,top=5)
print(results)




  





