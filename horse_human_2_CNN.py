import urllib.request
import zipfile
import numpy as np
from IPython.display import Image

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


def CNN_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')
    local_zip = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/validation-horse-or-human/')
    zip_ref.close()

    TRAINING_DIR = 'tmp/horse-or-human/'
    VALIDATION_DIR = 'tmp/validation-horse-or-human/'

    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        rescale=1 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary',
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary',
    )

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    checkpoint_path = "tmp_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)
    model.fit(train_generator,
              validation_data=(validation_generator),
              epochs=25,
              callbacks=[checkpoint],
              )

    model.load_weights(checkpoint_path)

    return model

model = CNN_model()