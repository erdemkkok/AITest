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
sicaklik=[0,0.25,0.5,1]
files=os.listdir('kap/')
files_1=os.listdir('sivi/')
print(files[0])
print(files_1[0])
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
labels=[1,1,0,0]
def oznitelikCikartici(inputer,modelNo):

    x = layers.Conv2D(32,(3,3),activation="relu",name=f"{modelNo}_1")(inputer)
    x = layers.Conv2D(32,(3,3),activation="relu",name=f"{modelNo}_2")(x)
    x = layers.Conv2D(64,(3,3),activation="relu",name=f"{modelNo}_3")(x)
    x = layers.Conv2D(64,(3,3),activation="relu",name=f"{modelNo}_4")(x)
    x = layers.Conv2D(128,(3,3),activation="relu",name=f"{modelNo}_5")(x)
    x = layers.Conv2D(128,(3,3),activation="relu",name=f"{modelNo}_6")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x=  layers.Flatten()(x)
    #x = layers.BatchNormalization()(x)
    return x

def sicaklikDense(inputer):
    x=layers.Dense(64)(inputer)
    x=layers.Dense(1)(x)
    return x

def benzirlikCikartici(inputer):
    
    x = layers.Dense(128,activation="relu")(inputer)
    #x = layers.Dense(64,activation="relu")(x)
    #x = layers.Dense(32,activation="relu")(x)
    x = layers.Dense(1,activation="sigmoid")(x)
    return x

def mimariDon():
    inputer1 = layers.Input(shape=(1,))
    inputer2 = layers.Input(shape=(32,32,3))
    inputer3 = layers.Input(shape=(32,32,3))
    m1 = sicaklikDense(inputer1)
    m2 = oznitelikCikartici(inputer2,"ana")
    m3=oznitelikCikartici(inputer3,"alt")
    birlesik = tf.keras.layers.add([m1, m2])
    birlesik = layers.Flatten()(birlesik)

    hesaplayici = benzirlikCikartici(birlesik)
    print(hesaplayici,m3)
    son_birlesik=tf.keras.layers.Concatenate(axis=1)([hesaplayici,m3])
    son_birlesik = layers.Flatten()(son_birlesik)
    hesaplayici=benzirlikCikartici(son_birlesik)
    model = Model(inputs = [inputer1,inputer2,inputer3],outputs=hesaplayici,name="benzerlikModeli")

    return model 

model = mimariDon()
model.summary()
#tf.keras.utils.plot_model(model=model,show_shapes=True,to_file='mymodel.png')
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


model.fit(x =[np.array(sicaklik),np.array(sivi),np.array(su_kap)],y = np.array(labels),batch_size=32,epochs=10,verbose=2)
# model.evaluate((np.array(su_kap),np.array(sivi)),np.array(labels))
print(model.predict([np.array(sicaklik[0]).reshape(1,1),np.array(su_kap[0]).reshape(1,32,32,3),np.array(sivi[1]).reshape(1,32,32,3)]))


"""binary_crossentropy loss fonksiyonu kullanılmıştır. binary olarak sınıflandırma yapılmıştır.
Dense çıkışı 1 olarak alınmıştır(Çünkü sadece True veya False şeklinde bir çıktı istenmektedir). 
Su kabi ve içine konulan cisimin sıcaklığa göre kabı delip delmeyeceği üzerinde bir çalışılmıştır.
"""