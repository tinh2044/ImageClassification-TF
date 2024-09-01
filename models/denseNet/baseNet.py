import keras
from keras import layers


def dense_conv2d(x, filters, kernel_size, strides=1, padding="valid", name=None):

    return keras.Sequential(
        [
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
        ], name=name)(x)


def dense_block(x, num_block, grown_rate, name=None):
    
    for i in range(num_block):
        name = f'{name}.{i + 1}'
        out = dense_conv2d(x, 4 * grown_rate, 1, name=f'{name}.conv1')
        out = dense_conv2d(out, grown_rate, 3, padding="same", name=f'{name}.conv2')

        x = layers.Concatenate(name=f'{name}.concat')([x, out])
    return x


def transition_block(x, reduction, name=None):
    x = layers.BatchNormalization(name=f"{name}.bn")(x)
    x = layers.ReLU(name=f"{name}.relu")(x)
    x = layers.Conv2D(int(x.shape[-1] * reduction), 1, name=f"{name}.conv2d")(x)

    return layers.AveragePooling2D(pool_size=(2, 2), strides=2, name=f"{name}.pool")(x)


def dense_net(num_cls, num_block, grown_rate=32, reduction=0.5, name=None):
    _input = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="conv1.conv")(_input)
    x = layers.BatchNormalization(name="conv1.bn")(x)
    x = layers.Activation("relu", name="conv.relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    for i in range(len(num_block)):
        x = dense_block(x, num_block[i], grown_rate, name=f"dense{i + 1}")
        x = transition_block(x, reduction, name=f"transition{i + 1}")
    
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    x = layers.Dense(num_cls, activation='softmax', name="output")(x)

    return keras.Model(inputs=_input, outputs=x, name=name)
