import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(25)

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)
    print("X_test original shape", X_test.shape)
    print("y_test original shape", y_test.shape)
    return X_train, y_train, X_test, y_test

def show(i, X_train, y_train):
    plt.imshow(X_train[i], cmap='gray')
    plt.title('Class '+ str(y_train[i]))
    
#(batches, height, widhth, channel)
def reshape(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    return X_train, X_test

#Here we’ve rescaled the image data so that each pixel lies in the interval [0, 1] instead of [0, 255]. It is always a good idea to normalize the input so that each dimension has approximately the same scale.
def rescale(X_train, X_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train/=255
    X_test/=255
    print('New shape of X_train is: ', X_train.shape)
    return X_train, X_test

def one_hot_encoding(i, y_train, y_test):
    number_of_classes = 10
    Y_train = np_utils.to_categorical(y_train, number_of_classes)
    Y_test = np_utils.to_categorical(y_test, number_of_classes)
    print('Before one hot encoding: ', y_train[i])
    print('After one hot encoding: ', Y_train[i])
    return Y_train, Y_test

def CNN_model(X_train, X_test, Y_train, Y_test, e):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10))

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    #to reduce overfitting : data augmentation
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

    test_gen = ImageDataGenerator()
    train_generator = gen.flow(X_train, Y_train, batch_size=64)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
    model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=e, 
                    validation_data=test_generator, validation_steps=10000//64)
    return model

def accuracy(X_test, Y_test, model):
    score = model.evaluate(X_test, Y_test)
    print()
    print('Test accuracy: ', score[1] * 100)

def main():
    X_train, y_train, X_test, y_test = load_data()
    i = 1
    show(i, X_train, y_train)
    X_train, X_test = reshape(X_train, X_test)
    X_train, X_test = rescale(X_train, X_test)
    Y_train, Y_test = one_hot_encoding(i, y_train, y_test)
    # epochs = 3
    e = 2
    model = CNN_model(X_train, X_test, Y_train, Y_test, e)
    accuracy(X_test, Y_test, model)

main()    