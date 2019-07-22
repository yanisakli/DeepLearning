from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from keras.preprocessing import image
import numpy as np
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import tensorflow as tf

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


#tensorboard = TensorBoard(log_dir="./weka".format(time()))

trainDatagen = ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

testDatagen = ImageDataGenerator(rescale=1. / 255)

training_set = trainDatagen.flow_from_directory('./dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = testDatagen.flow_from_directory('./dataset/test_set',
                                           target_size=(64, 64),
                                           batch_size=32,
                                           class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8,
                         epochs=2,
                         #callbacks=[tensorboard],
                         validation_data=test_set,
                         max_queue_size=2000)


test_image = image.load_img('chat1.png',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
print(training_set.class_indices)

#with tf.compat.v1.Session() as sess :
 #   writer = tf.compat.v1.summary.FileWriter('./weka', sess.graph)


now=datetime.datetime.now()

#save the training
#classifier.save(open("Model"+now.isoformat(),"wb"))
#classifier.save('/Users/mac/Desktop/DeepLearning/DeepLearning/venv/project.h5')
#test the training : load
img = image.load_img('chat2.png',target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

mdl = load_model("Model1")

print(mdl.predict(img))
print(training_set.class_indices)