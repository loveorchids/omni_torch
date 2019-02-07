try:
    import keras
except ImportError:
    raise ImportError("You need to download keras>=2.2.0")
from keras.models import Sequential
from keras.layers import *
from keras.datasets import *
from keras.preprocessing.image import ImageDataGenerator
import os

save_dir = os.path.join(os.getcwd(), 'models')
model_name = 'cifar10_cnn.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train)  # (50000,32,32,3)
# print(y_train)  # (50000, 1)
# print(x_test)  # (10000,32,32,3)
# print(y_test)  # (10000,1)

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
# exit(1)
opt = keras.optimizers.adam(lr=1e-4, decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

#data_augmentation = True
data_augmentation = False
if not data_augmentation:
    print('Not using data augmentation')
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), shuffle=True)
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
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), epochs=100, validation_data=(x_test, y_test), workers=4)  # 之前的keras必须要steps_per_epoch，workers：最大进程数

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

scores = model.evaluate(x_test, y_test)
print(scores)
