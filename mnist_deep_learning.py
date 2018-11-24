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

print(X_train.shape)
print(X_test.shape)
