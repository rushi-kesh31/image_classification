#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[2]:


import os
import tensorflow as tf
from tensorflow import keras 


# In[3]:


train_dir=r'C:\Users\91898\Downloads\cats_and_dogs_small\train'
valid_dir=r'C:\Users\91898\Downloads\cats_and_dogs_small\validation'
test_dir=r'C:\Users\91898\Downloads\cats_and_dogs_small\test'


# In[4]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


train_datagen=ImageDataGenerator(rescale=1./255,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,zoom_range=0.2,rotation_range=40,shear_range=0.2)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_dir,batch_size=20,target_size=(150,150),class_mode='binary')
valid_generator=test_datagen.flow_from_directory(valid_dir,batch_size=20,target_size=(150,150),class_mode='binary')


# In[6]:


from tensorflow.keras.applications import VGG16 


# In[7]:


conv_base=VGG16(weights="imagenet",include_top=False,input_shape=(150,150,3))


# In[8]:


from tensorflow.keras import layers
from tensorflow.keras import models


# In[9]:


model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))


# In[10]:


model.summary()


# In[11]:


from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=2e-5),metrics=["accuracy"])


# In[13]:


model_history=model.fit(train_generator,validation_data=valid_generator,validation_steps=50,epochs=20,steps_per_epoch=100)



