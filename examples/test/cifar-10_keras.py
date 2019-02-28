"""
# Copyright (c) 2018 Works Applications Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
import keras
from keras.models import Sequential
from keras.layers import *
from keras.datasets import *
from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs=None):
        print(K.eval(self.model.optimizer.lr))
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return

histories = Histories()
save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'cifar10_cnn.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255

model = Sequential()
model.add(Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#model.summary()

opt = keras.optimizers.adam(lr=1e-4, decay=1e-6, epsilon=1e-7)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt,
              metrics=['accuracy'])

model_path = os.path.join(os.getcwd(), 'models', "cifar10_cnn.h5")
print("load model weight from: %s"%(model_path))
model.load_weights(model_path)

#data_augmentation = True
data_augmentation = False
if not data_augmentation:
    print('Not using data augmentation')
    model.fit(x_train, y_train, batch_size=256, epochs=200, validation_data=(x_test, y_test),
              shuffle=False, callbacks=[histories])
else:
    print('Using real-time data augmentation')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=100,
                        validation_data=(x_test, y_test), workers=4)

"""
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir, model_name)
print("Save model to: %s"%(model_path))
model.save(model_path)
"""

scores = model.evaluate(x_test, y_test)
print(scores)
