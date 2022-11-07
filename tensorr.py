import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images[0].shape)       
train_images = train_images / 255.0
test_images = test_images / 255.0
# plt.figure()
# print("LABEL for SHOES",train_labels[0])
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.add(tf.keras.layers.Dropout(0.2))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(type(train_images),type(train_labels))
model.fit(train_images, train_labels, epochs=10)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
print(test_labels[0])
print(np.argmax(predictions[0]))