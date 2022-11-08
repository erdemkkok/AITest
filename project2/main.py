from sklearn.utils import shuffle
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
su_kaplari=[]
sivilar=[]
files=os.listdir('kap/')
files_1=os.listdir('sivi/')
print(files[0])
print(files_1[0])
veriler=[]
#for i in range(len(files)):
for i in range(len(files)):
    img=cv2.imread("kap/"+files[i])
    img=cv2.resize(img,(32,32))
    su_kaplari.append(img)
for i in range(len(files_1)):
    img1=cv2.imread("sivi/"+files_1[i])
    img1=cv2.resize(img1,(32,32))
    sivilar.append(img1)
su_kap=[su_kaplari[0],su_kaplari[1],su_kaplari[0],su_kaplari[1]]
sivi=[sivilar[0],sivilar[1],sivilar[1],sivilar[0]]
labels=[0,1,1,0]
def oznitelikCikartici(inputer,modelNo):

    x = layers.Conv2D(32,(3,3),activation="relu",name=f"{modelNo}_1")(inputer)
    x = layers.Conv2D(32,(3,3),activation="relu",name=f"{modelNo}_2")(x)
    x = layers.Conv2D(64,(3,3),activation="relu",name=f"{modelNo}_3")(x)
    x = layers.Conv2D(64,(3,3),activation="relu",name=f"{modelNo}_4")(x)
    x = layers.Conv2D(128,(3,3),activation="relu",name=f"{modelNo}_5")(x)
    x = layers.Conv2D(128,(3,3),activation="relu",name=f"{modelNo}_6")(x)
    x = layers.MaxPooling2D((2,2))(x)
    #x = layers.BatchNormalization()(x)
    return x

def benzirlikCikartici(inputer):
    
    x = layers.Dense(128,activation="relu")(inputer)
    #x = layers.Dense(64,activation="relu")(x)
    #x = layers.Dense(32,activation="relu")(x)
    x = layers.Dense(2,activation="sigmoid")(x)
    return x

def mimariDon():
    inputer1 = layers.Input(shape=(32,32,3))
    inputer2 = layers.Input(shape=(32,32,3))
    m1 = oznitelikCikartici(inputer1,"ana")
    m2 = oznitelikCikartici(inputer2,"alt")
    
    birlesik = tf.keras.layers.concatenate([m1, m2], axis=1)
    birlesik = layers.Flatten()(birlesik)

    hesaplayici = benzirlikCikartici(birlesik)

    model = Model(inputs = [inputer1,inputer2],outputs=hesaplayici,name="benzerlikModeli")

    return model 

model = mimariDon()
model.summary()
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])


model.fit(x =[np.array(su_kap),np.array(sivi)],y = np.array(labels),batch_size=32,epochs=10,verbose=2)
model.evaluate((np.array(su_kap),np.array(sivi)),np.array(labels))
print(model.predict([np.array(su_kap[0]).reshape(1,32,32,3),np.array(sivi[1]).reshape(1,32,32,3)]))


"""sparse_categorical_crossentropy loss fonksiyonu kullanılmıştır. kategorik olarak sınıflandırma yapılmıştır.
Dense çıkışı 2 olarak alınmıştır. Burada mantıksal olarak doğruluk önemli bir noktadır.Yoksa Dense Çıkışı 5 olarakta verilebilir.
Su kabi ve içine konulan cisimin kabı delip delmeyeceği üzerinde bir çalışılmıştır.

"""