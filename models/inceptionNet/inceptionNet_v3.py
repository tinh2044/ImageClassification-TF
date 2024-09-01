import keras
from keras import layers


def conv2d(x, filters, kernel_size, strides=1, padding='valid', name=None):
   
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
        layers.BatchNormalization(),
        layers.Activation('relu'),
    ], name=name)(x)


def inception_a(x, _filter1x1, middle3x3, _filter3x3, middle3x3_dbl, _filter3x3_dbl, _filter_pool, name):
    
    branch_1x1 = conv2d(x, 64, (1, 1), name=f'{name}.1x1')

    branch_3x3 = conv2d(x, middle3x3, (1, 1), name=f'{name}.middle_3x3')
    branch_3x3 = conv2d(branch_3x3, _filter3x3, (3, 3), padding='same', name=f'{name}.3x3')

    branch_3x3_dbl = conv2d(x, middle3x3_dbl, (1, 1), name=f'{name}.middle_3x3_dbl')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 3), padding='same', name=f'{name}.3x3_dbl_1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 3), padding='same', name=f'{name}.3x3_dbl_2')

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, _filter_pool, (1, 1), name=f'{name}.pool')

    return layers.Concatenate(name=f"{name}.concat")([branch_1x1, branch_3x3, branch_3x3_dbl, branch_pool])


def reduce_inception_a(x, name):

    branch_3x3 = conv2d(x, 384, (3, 3), strides=2, name=f'{name}.3x3')

    branch_3x3_dbl = conv2d(x, 64, 1, 1, name=f'{name}.1x1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, 96, (3, 3), padding="same", name=f'{name}.3x3_1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, 96, (3, 3), strides=2, name=f'{name}.3x3_2')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), name=f"{name}.pool")(x)

    return layers.Concatenate(name=f"{name}.concat")([branch_3x3, branch_3x3_dbl, branch_pool])


def inception_b(x, _filter1x1, middle7x7, _filter7x7, middle7x7_dbl, _filter7x7_dbl, _filter_pool, name):
    
    branch_1x1 = conv2d(x, 192, (1, 1), name=f'{name}.1x1')

    branch_7x7 = conv2d(x, middle7x7, (1, 1), name=f'{name}.middle_7x7')
    branch_7x7 = conv2d(branch_7x7, middle7x7, (1, 7), padding="same", name=f'{name}.1x7')
    branch_7x7 = conv2d(branch_7x7, _filter7x7, (7, 1), padding="same", name=f'{name}.7x1')

    branch_7x7_dlb = conv2d(x, middle7x7_dbl, (1, 1), name=f'{name}.middle_7x7_dbl')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, middle7x7_dbl, (1, 7), padding="same", name=f'{name}.1x7_1')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, middle7x7_dbl, (7, 1), padding="same", name=f'{name}.7x1_1')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, middle7x7_dbl, (1, 7), padding="same", name=f'{name}.1x7_2')
    branch_7x7_dlb = conv2d(branch_7x7_dlb, _filter7x7_dbl, (7, 1), padding="same", name=f'{name}.7x1_2')

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d(branch_pool, _filter_pool, (1, 1), name=f'{name}.pool')

    return layers.Concatenate(name=f"{name}.concat")([branch_1x1, branch_7x7, branch_7x7_dlb, branch_pool])


def reduce_inception_b(x, name):
    branch_3x3 = conv2d(x, 192, (1, 1), name=f"{name}.middel_3x3")
    branch_3x3 = conv2d(branch_3x3, 320, (3, 3), 2, name=f"{name}.3x3")

    branch_7x7x3 = conv2d(x, 192, (1, 1), name=f"{name}.1x1")
    branch_7x7x3 = conv2d(branch_7x7x3, 192, (1, 7), name=f"{name}.1x7", padding="same")
    branch_7x7x3 = conv2d(branch_7x7x3, 192, (7, 1), name=f"{name}.7x1", padding="same")
    branch_7x7x3 = conv2d(branch_7x7x3, 192, (3, 3), strides=2, name=f"{name}.3x3_1")

    branch_pool = layers.MaxPooling2D((3, 3), (2, 2), name=f"{name}.pool")(x)

    return layers.Concatenate(name=f"{name}.concat")([branch_3x3, branch_7x7x3, branch_pool])


def inception_c(x, _filter1x1, _filter3x3, _middle_3x3_dbl, _filter3x3_dbl, _filter_pool, name):
    
    branch_1x1 = conv2d(x, _filter1x1, (1, 1), name=f'{name}.1x1')

    branch_3x3 = conv2d(x, _filter3x3, (1, 1), name=f'{name}.middle_1x1')
    branch_3x3_1x3 = conv2d(branch_3x3, _filter3x3, (1, 3), padding="same", name=f'{name}.1x3')
    branch_3x3_3x1 = conv2d(branch_3x3, _filter3x3, (3, 1), padding="same", name=f'{name}.3x1')
    branch_3x3 = layers.Concatenate(name=f"{name}.concat_3x3")([branch_3x3_1x3, branch_3x3_3x1])

    branch_3x3_dbl = conv2d(x, _middle_3x3_dbl, (1, 1), name=f'{name}.dbl_1x1')
    branch_3x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 3), padding="same", name=f'{name}.dbl_3x3')
    branch_3x3_1x3_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (1, 3), padding="same", name=f"{name}.dbl_1x3")
    branch_3x3_3x1_dbl = conv2d(branch_3x3_dbl, _filter3x3_dbl, (3, 1), padding="same", name=f"{name}.dbl_3x1")
    branch_3x3_dbl = layers.Concatenate(name=f"{name}.concat_3x3_dbl")([branch_3x3_1x3_dbl, branch_3x3_3x1_dbl])

    branch_pool = layers.AveragePooling2D((3, 3), (1, 1), padding="same", name=f"{name}.avg_pool")(x)
    branch_pool = conv2d(branch_pool, _filter_pool, (1, 1), name=f"{name}.conv_pool")

    return layers.Concatenate(name=f"{name}.concat")([branch_1x1, branch_3x3, branch_3x3_dbl, branch_pool])


def InceptionNetV3(num_cls):
    
    _input = layers.Input(shape=(299, 299, 3), name="input")

    x = conv2d(_input, 32, (3, 3), strides=2, name='conv_1')
    x = conv2d(x, 32, (3, 3), name='conv_2')
    x = conv2d(x, 64, (3, 3), strides=2, name='conv_3')
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same", name='avg_pool_1')(x)

    x = conv2d(x, 80, (1, 1), name='conv_4')
    x = conv2d(x, 192, (3, 3), name='conv_5')

    x = inception_a(x, 64, 48, 64, 64, 96, 32, name="inception_a1")
    x = inception_a(x, 64, 48, 64, 64, 96, 64, name="inception_a2")
    x = inception_a(x, 64, 48, 64, 64, 96, 64, name="inception_a3")

    x = reduce_inception_a(x, name="reduce_a")

    x = inception_b(x, 192, 128, 192, 128, 192, 192, name="inception_b1")
    x = inception_b(x, 192, 160, 192, 160, 192, 192, name="inception_b2")
    x = inception_b(x, 192, 160, 192, 160, 192, 192, name="inception_b3")
    x = inception_b(x, 192, 192, 192, 192, 192, 192, name="inception_b4")

    x = reduce_inception_b(x, name="reduce_b")

    x = inception_c(x, 320, 384, 448, 384, 192, name="inception_c1")
    x = inception_c(x, 320, 384, 448, 384, 192, name="inception_c2")

    x = layers.GlobalAveragePooling2D(name="avg_pool_2")(x)
    x = layers.Dense(num_cls, activation='softmax', name="output")(x)
    return keras.Model(inputs=_input, outputs=x, name='InceptionNetV3')

