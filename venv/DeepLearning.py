from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *


classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

trainDatagen = ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

testDatagen = ImageDataGenerator(rescale=1. / 255)

training_set = trainDatagen.flow_from_directory('/Users/mac/Desktop/Convolutional_Neural_Networks/dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = testDatagen.flow_from_directory('/Users/mac/Desktop/Convolutional_Neural_Networks/dataset/test_set',
                                           target_size=(64, 64),
                                           batch_size=32,
                                           class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=10,
                         epochs=2,
                         validation_data=test_set,
                         max_queue_size=2000)