"""
ELEN0016-2 - Computer vision
University of Liège
Academic year 2019-2020

Student project - Part 2
Sudoku digit recognition and performance assessment
"""

#############
# Libraries #
#############

import os
import cv2
import numpy as np

from sudoku import threshold

from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


###########
# Classes #
###########

class DigitClassifier():
    '''Abstract digit classifier.'''  
    def __init__(self, warm_start=None):
        if warm_start is None:
            self.model = None
        else:
            self.model = load_model(warm_start)

    def preprocessing(self, img):
        '''Preprocesses image.'''
        img = threshold(img, n_block=2, c=20)
        img = cv2.resize(img, (28, 28))
        cv2.rectangle(img, (-1, -1), img.shape, 0, thickness=2)
        img = img.reshape(1, 28, 28, 1).astype('float32') / 255

        return img

    def predict_proba(self, img):
        '''Predicts classes probabilities.'''
        return self.model.predict_proba(self.preprocessing(img))

    def predict(self, img):
        '''Predicts class.'''
        proba = self.predict_proba(img)[0]

        y = np.argmax(proba)
        return y if proba[y] > 0.55 else 0

    def save(self, filename):
        '''Saves model.'''
        self.model.save(filename)


class Ghouzam(DigitClassifier):
    '''Ghouzam digit classifier. Source : https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6?fbclid=IwAR12dON7-HGwZrjwan9H8UUaftW3Jk7nyv6mJuQ6keSr9yPEYEMwsQhQLdg'''
    def __init__(self, warm_start=None, epochs=10, batch_size=86):
        super().__init__(warm_start)

        self.epochs = epochs
        self.batch_size = batch_size

        if self.model is None:
            self.model = Sequential()

            self.model.add(Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding='Same',
                activation='relu',
                input_shape=(28,28,1)
            ))
            self.model.add(Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding='Same',
                activation ='relu'
            ))
            self.model.add(MaxPool2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.25))
            self.model.add(Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='Same',
                activation='relu'
            ))
            self.model.add(Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding='Same',
                activation='relu'
            ))
            self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Dropout(0.25))

            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(10, activation='softmax'))

            # Optimizer
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

            # Compile model
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Data augmentation to prevent overfitting
            self.datagen = ImageDataGenerator(
                featurewise_center=False, # set input mean to 0 over the dataset
                samplewise_center=False, # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False, # apply ZCA whitening
                rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range=0.1, # Randomly zoom image 
                width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
                horizontal_flip=False, # randomly flip images
                vertical_flip=False # randomly flip images
            )

    def fit(self, X_LS, y_LS, X_VS, y_VS):
        '''Fits model.'''

        # One-hot encoding
        y_LS = to_categorical(y_LS)
        y_VS = to_categorical(y_VS)

        # Fit
        self.datagen.fit(X_LS)

        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        )

        self.model.fit_generator(
            self.datagen.flow(X_LS, y_LS, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(X_VS, y_VS),
            verbose=2,
            steps_per_epoch=X_LS.shape[0] // self.batch_size,
            callbacks=[learning_rate_reduction]
        )

        return self


class MLM(DigitClassifier):
    '''MLM digit classifier. Source : https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/'''
    def __init__(self, warm_start=None, epochs=30, batch_size=86):
        super().__init__(warm_start)

        self.epochs = epochs
        self.batch_size = batch_size

        if self.model is None:
            self.model = Sequential()

            self.model.add(Conv2D(
                filters=30,
                kernel_size=(5, 5),
                input_shape=(28, 28, 1),
                activation='relu'
            ))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(15, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(50, activation='relu'))
            self.model.add(Dense(10, activation='softmax'))

            # Compile model
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

    def fit(self, X_LS, y_LS, X_VS, y_VS):
        '''Fits model.'''

        # One-hot encoding
        y_LS = to_categorical(y_LS)
        y_VS = to_categorical(y_VS)

        # Fit
        self.model.fit(
            X_LS,
            y_LS,
            validation_data=(X_VS, y_VS),
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        return self


class Gkoehler(DigitClassifier):
    '''MLM digit classifier. Source : https://nextjournal.com/gkoehler/digit-recognition-with-keras'''
    def __init__(self, warm_start=None, epochs=20, batch_size=128):
        super().__init__(warm_start)

        self.epochs = epochs
        self.batch_size = batch_size

        if self.model is None:
            self.model = Sequential()

            self.model.add(Dense(512, input_shape=(28 * 28,), activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(10, activation='softmax'))

            # Compile model
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

    def fit(self, X_LS, y_LS, X_VS, y_VS):
        '''Fits model.'''

        # Reshape
        X_LS = X_LS.reshape(-1, 28 * 28)
        X_VS = X_VS.reshape(-1, 28 * 28)

        # One-hot encoding
        y_LS = to_categorical(y_LS)
        y_VS = to_categorical(y_VS)

        # Fit
        self.model.fit(
            X_LS,
            y_LS,
            validation_data=(X_VS, y_VS),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2
        )

        return self

    def preprocessing(self, img):
        '''Preprocesses image.'''
        img = super().preprocessing(img)
        img = img.reshape(1, 28 * 28)

        return img


##############
# Parameters #
##############

# Model
MODEL = Ghouzam()

# Model destination
DESTINATION = '../products/models/'

# Model filename
FILENAME = 'ghouzam.h5'


########
# Main #
########

if __name__ == '__main__':
    from keras.datasets import mnist

    # mkdir -p DESTINATION
    os.makedirs(DESTINATION, exist_ok=True)

    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Fit
    MODEL.fit(X_train, y_train, X_test, y_test)
    MODEL.save(DESTINATION + FILENAME)
