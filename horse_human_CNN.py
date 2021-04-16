import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


dataset_name = 'horses_or_humans'

train_dataset = tfds.load(name=dataset_name, split='train[:80%]') # data의 80%
valid_dataset = tfds.load(name=dataset_name, split='train[80%:]') # data의 20%


def preprocess(data):
    x = data['image']
    y = data['label']
    # Normalisation
    x = tf.cast(x, tf.float32) / 255.0
    # (224, 224)
    x = tf.image.resize(x, size=(224, 224))
    return x, y


def CNN_model():
    batch_size = 32
    train_data = train_dataset.map(preprocess).batch(batch_size)
    valid_data = valid_dataset.map(preprocess).batch(batch_size)

    model = Sequential([
        Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),

        # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    checkpoint_path = "my_checkpoint.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    model.fit(train_data,
              validation_data=(valid_data),
              epochs=20,
              callbacks=[checkpoint],
              )

    model.load_weights(checkpoint_path)

    return model

model = CNN_model()