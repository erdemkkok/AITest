import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
files=os.listdir('fare/')
files_1=os.listdir('klavye/')
print(files[0])
veriler=[]
for i in range(len(files)):
    img=cv2.imread("fare/"+files[i])
    img=cv2.resize(img,(32,32))
    veriler.append(img)
for i in range(len(files_1)):
    img1=cv2.imread("klavye/"+files_1[i])
    img1=cv2.resize(img1,(32,32))
    veriler.append(img1)

labels=[0,0,0,1,1,1]


model=models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(veriler), np.array(labels), epochs=10)
model=models.Sequential([model,layers.Activation(tf.keras.activations.sigmoid)])
print(model.predict(veriler[4].reshape(1,32,32,3)))