import keras
from keras.api.layers import Conv2D, ReLU, MaxPooling2D, Concatenate, AvgPool2D, Dense, Dropout, \
    Flatten, GlobalAveragePooling2D
from keras import Model, Sequential, Input


def conv_block(x, _filter: int,

               kernel_size: int,
               strides: int,
               padding: str,
               name: str) -> Model:
    
    return Sequential(
        [Conv2D(_filter, kernel_size, strides, padding=padding),
         ReLU(), ],
        name=name)(x)


def inception_block(x,
                    filter_1,
                    middel_3, filter_3,
                    middel_5, filter_5,
                    filter_last, name) -> Model:
    conv_1x1 = conv_block(x, filter_1, kernel_size=1, strides=1, padding='same', name=f"{name}.conv_1x1")

    middle_3x3 = conv_block(x, middel_3, kernel_size=1, strides=1, padding="same", name=f"{name}.middel_3x3")
    conv_3x3 = conv_block(middle_3x3, filter_3, kernel_size=3, strides=1, padding="same", name=f"{name}.conv_3x3")

    middle_5x5 = conv_block(x, middel_5, kernel_size=1, strides=1, padding="same", name=f"{name}.middel_5x5")
    conv_5x5 = conv_block(middle_5x5, filter_5, kernel_size=3, strides=1, padding="same", name=f"{name}conv_5x5")

    conv_pool = MaxPooling2D(pool_size=3, strides=1, padding="same", name=f"{name}.pool")(x)
    conv_pool = conv_block(conv_pool, filter_last, kernel_size=1, strides=1, padding="same", name=f"{name}.conv_last")

    return Concatenate(name=f"{name}.concat")([conv_1x1, conv_3x3, conv_5x5, conv_pool])


def auxiliary_block(x, num_cls, name):
    
    pool = AvgPool2D(pool_size=(5, 5), strides=(3, 3), name=f"{name}.avg_pool")(x)
    conv = conv_block(pool, 128, kernel_size=1, strides=1, padding="valid", name=f"{name}.conv")

    classifier = Sequential(
        [
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(num_cls, activation="softmax")
        ], name=f"{name}.classifier"
    )(conv)

    return classifier


def InceptionNetV1(num_cls):
    
    _input = Input(shape=(224, 224, 3), name="input")
    layer_1 = Sequential(
        [
            Conv2D(64, kernel_size=7, strides=2, padding="same", name="layer1.conv1"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            Conv2D(192, kernel_size=3, strides=1, padding="same", name="layer1.conv2"),
            Conv2D(192, kernel_size=3, strides=1, padding="same", name="layer1.conv3"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")],
        name="layer_1",
    )(_input)
    inception_3a = inception_block(layer_1, 64, 96, 128, 16, 32, 32, name='inception_3a')
    inception_3b = inception_block(inception_3a, 128, 128, 192, 32, 96, 64, name='inception_3b')

    pool_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool_4")(inception_3b)

    inception_4a = inception_block(pool_4, 192, 96, 208, 16, 48, 64, name='inception_4a')
    inception_4b = inception_block(inception_4a, 160, 112, 224, 24, 64, 64, name='inception_4b')
    inception_4c = inception_block(inception_4b, 128, 128, 256, 24, 64, 64, name='inception_4c')
    inception_4d = inception_block(inception_4c, 256, 160, 320, 32, 128, 128, name='inception_4d')

    pool_5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool_5")(inception_4d)

    inception_5a = inception_block(pool_5, 256, 160, 320, 32, 128, 128, name='inception_5a')
    inception_5b = inception_block(inception_5a, 384, 192, 384, 48, 128, 128, name='inception_5b')

    auxiliary_4a = auxiliary_block(inception_4a, num_cls, name="auxiliary_4a")
    auxiliary_4d = auxiliary_block(inception_4d, num_cls, name="auxiliary_4d")

    avg_pool = GlobalAveragePooling2D()(inception_5b)
    dropout = Dropout(0.5)(avg_pool)
    output = Dense(num_cls, activation="softmax")(dropout)

    return Model(inputs=_input, outputs=output, name="InceptionNetV1")
