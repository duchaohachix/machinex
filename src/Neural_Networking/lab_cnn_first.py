import tensorflow as tf
import numpy as np
import json
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()
model = Sequential(
    [
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #conv
        layers.MaxPooling2D((2, 2)), # pooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), # pooling
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), # pooling
        #output
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ]
)
model.summary()
## config model
## about adam https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
## train model
history = model.fit(train_images, train_labels, epochs=3, 
                     validation_data=(test_images, test_labels))
label_test = np.argmax(model.predict(test_images[:10]) , axis =1)

for i in range(10):
    print(f"Predict: {class_names[label_test[i]]} , label ban đầu: {class_names[test_labels[i][0]]} " )

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[label_test[i]])
plt.show()