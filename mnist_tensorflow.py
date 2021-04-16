import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def mnist_model():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0
    # YOUR CODE HERE

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        # Dense Layer
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax'),
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                save_weights_only=True,
                                save_best_only=True,
                                monitor='val_loss',
                                verbose=1)

    model.fit(x_train, y_train,
                        validation_data=(x_valid, y_valid),
                        epochs=20,
                        callbacks=[checkpoint],
                      )

    model.load_weights(checkpoint_path)



    return model

model = mnist_model()