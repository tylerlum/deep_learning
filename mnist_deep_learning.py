import numpy as np
import matplotlib.pyplot as plt
import random

## Keras
import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

def create_model(num_pixels, num_classes):
    model = Sequential()
    model.add(Dense(units=10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

np.random.seed(0)

## Get mnist training data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

## Check the sizing
assert(X_train.shape[0] == Y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[0] == Y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28x28" 

## Setup 
num_of_samples = []
cols = 5
num_classes = 10

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10,10))
fig.tight_layout()

## Show images of numbers
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[Y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis('off')        

        ## Label each row in the middle
        if i == int(cols/2):
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
plt.show()

## Plot sizes of images
plt.figure(figsize=(12,4))
plt.bar(range(0,num_classes), num_of_samples)
plt.title('Distribution of the training dataset')
plt.xlabel('Class number')
plt.ylabel('Number of images')
plt.show()

## One hot encoding
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

## Normalize data down 
X_train = X_train / 255
X_test= X_test / 255

## Flatten images
num_pixels = len(X_train[1]) * len(X_train[2])
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

## Create neural network
model = create_model(num_pixels, num_classes)
history = model.fit(X_train, Y_train, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle='true')

## Plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('acc')
plt.xlabel('epoch')
plt.show()

## Test model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score {0}'.format(score[0]))
print('Test accuracy {0}'.format(score[1]))

