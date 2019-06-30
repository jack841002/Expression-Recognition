#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import cv2


# In[2]:


image_size = 100
path_name = './expression/original'


# In[3]:


images = []
labels = []
path = []
imageID = []
ID = []


# In[4]:


def resize_image(image, height = image_size, width = image_size):
    top, bottom, left, right = (0, 0, 0, 0)
     
    #獲取影象尺寸
    h, w, _ = image.shape
    
    #對於長寬不相等的圖片，找到最長的一邊
    longest_edge = max(h, w)    
    
    #計算短邊需要增加多上畫素寬度使其與長邊等長
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    #RGB顏色
    BLACK = [0, 0, 0]
    
    #給影象增加邊界，是圖片長、寬等長，cv2.BORDER_CONSTANT指定邊界顏色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    #調整影象大小並返回
    return cv2.resize(constant, (height, width))


# In[5]:


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        #從初始路徑開始疊加，合併成可識別的操作路徑
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):    #如果是資料夾，繼續遞迴呼叫
            read_path(full_path)
        else: #檔案
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, image_size, image_size)
                
                #放開這個程式碼，可以看到resize_image()函式的實際呼叫效果
                #cv2.imwrite('1.jpg', image)
                
                images.append(image)  
                path.append(full_path)                                
                
    return images, path


# In[6]:


images, path = read_path(path_name)


# In[7]:


for i in range(len(path)):
    ID.append(path[i].split('\\')[-1])


# In[8]:


print(ID)


# In[9]:


with open('expression/list_patition_label1.txt', 'r') as f:
    label = f.read().splitlines()

for i in range(len(label)):
    imageID.append(label[i].split(' ')[0])
    labels.append(label[i].split(' ')[1])


# In[10]:


newlabels = []
for i in ID:
    index = imageID.index(i)
    newlabels.append(labels[index])


# In[11]:


print(newlabels)


# In[12]:


#將輸入的所有圖片轉成四維陣列，尺寸為(圖片數量*IMAGE_SIZE*IMAGE_SIZE*3)
#我和閨女兩個人共1200張圖片，IMAGE_SIZE為64，故對我來說尺寸為1200 * 64 * 64 * 3
#圖片為64 * 64畫素,一個畫素3個顏色值(RGB)
images = np.array(images)
print(images.shape) 


# In[13]:


newlabels = np.array(newlabels)
print(newlabels.shape)


# In[ ]:





# In[14]:


import random

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

#from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE


# In[15]:


temp_images, test_images, temp_labels, test_labels = train_test_split(images, newlabels, test_size=0.1,random_state = random.randint(0, 100))


# In[16]:


train_images, valid_images, train_labels, valid_labels = train_test_split(temp_images, temp_labels, test_size=0.2, random_state = random.randint(0, 100))


# In[17]:


#把label的1~11改成0~10，為了作one hot encoding
for i in range(len(train_labels)):
    train_labels[i] = np.int(train_labels[i]) - 1

for i in range(len(valid_labels)):
    valid_labels[i] = np.int(valid_labels[i]) - 1
    
for i in range(len(test_labels)):
    test_labels[i] = np.int(test_labels[i]) - 1


# In[18]:


#我們的模型使用categorical_crossentropy作為損失函式，因此需要根據類別數量nb_classes
#類別標籤進行one-hot編碼使其向量化，在這裡我們的類別只有兩種，經過轉化後標籤資料變為二維

classes = 11

train_labels = np_utils.to_categorical(train_labels, classes)
valid_labels = np_utils.to_categorical(valid_labels, classes)
test_labels = np_utils.to_categorical(test_labels, classes)


# In[19]:


#畫素資料浮點化以便歸一化
train_images = train_images.astype('float32') 
valid_images = valid_images.astype('float32')
test_images = test_images.astype('float32')


# In[20]:


#將其歸一化,影象的各畫素值歸一化到0~1區間
train_images /= 255
valid_images /= 255
test_images /= 255


# In[22]:


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = (100,100,3) ))  #卷積層
model.add(Activation('relu'))  #啟用函式層
model.add(MaxPooling2D(2,2))  #池化層
model.add(Dropout(0.25))  #dropout層

model.add(Convolution2D(50, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25)) 

model.add(Flatten())   #flatten層

model.add(Dense(512))   #全聯通層
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(classes))
model.add(Activation('softmax'))   #分類層

model.summary()


# In[23]:


#訓練模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


model.fit(x=train_images,y=train_labels,validation_split=0.2, epochs=10, batch_size=300,verbose=2)


# In[25]:


result = model.evaluate(train_images, train_labels)
print ('\ntrain Acc:', result[1])


# In[26]:


result = model.evaluate(test_images, test_labels)
print ('\nTest Acc:', result[1])


# In[ ]:




