import keras
from keras import layers


def VGG16(num_cls):
    _input = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1.conv1"
    )(_input)
    x = layers.Conv2D(
        64, (3, 3), activation="relu", padding="same", name="block1.conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1.pool")(x)

    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2.conv1"
    )(x)
    x = layers.Conv2D(
        128, (3, 3), activation="relu", padding="same", name="block2.conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2.pool")(x)

    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3.conv1"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3.conv2"
    )(x)
    x = layers.Conv2D(
        256, (3, 3), activation="relu", padding="same", name="block3.conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3.pool")(x)

    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4.conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4.conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block4.conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4.pool")(x)

    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5.conv1"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5.conv2"
    )(x)
    x = layers.Conv2D(
        512, (3, 3), activation="relu", padding="same", name="block5.conv3"
    )(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5.pool")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, activation="relu", name="fc1")(x)
    x = layers.Dense(4096, activation="relu", name="fc2")(x)

    x = layers.Dense(num_cls, activation="softmax", name="output")(x)

    return keras.Model(inputs=_input, outputs=x, name="VGG16")


if __name__ == "__main__":
    model = VGG16(10)
    model.summary()