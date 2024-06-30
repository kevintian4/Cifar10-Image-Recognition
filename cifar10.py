#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# ## Import cifar10 dataset and check data shape

# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train.shape


# ## Normalize data and verify

# In[3]:


x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)
# x_train[0][0]


# ## Import Sequential model, layers, and Adam optimizer

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2, L1


# ## Build model and compile

# In[5]:


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# ## Train model

# In[6]:


model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


# ## Create an array to map integer representations to object name

# In[7]:


mapping = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# ## Duplicate original y_test dataset to display original images

# In[8]:


(x_train2, y_train2), (x_test2, y_test2) = tf.keras.datasets.cifar10.load_data()
# plt.imshow(x_test2[15])


# ## Write a function for verifying predictions

# In[9]:


def verify(original_images, labels, index):
    plt.figure(figsize=(2,2))
    plt.imshow(original_images[index])  # Display the image using matplotlib
    plt.title(f"Image Prediction: {mapping[np.argmax(labels[index])]}")  # Add the corresponding prediction as the title
    plt.xlabel(f"Actual: {mapping[y_test2[index][0]]}")  # Add the actual class of the image as the x-axis label
    plt.show()


# In[10]:


predictions = model.predict(x_test)
verify(x_test2, predictions, 20)


# In[11]:


for i in range(0, 10):
    verify(x_test2, predictions, i)

model.evaluate(x_test, y_test)

model.save("cifar10_model.h5")