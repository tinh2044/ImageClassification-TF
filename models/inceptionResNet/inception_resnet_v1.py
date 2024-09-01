from keras import layers
import keras


def conv2d(x, filters, kernel_size, strides=1, padding='valid', name=None):
    
    return keras.Sequential([
        layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
        layers.BatchNormalization(),
        layers.Activation('relu'),
    ], name=name)(x)


def inception_res_a(x, scale, name):
    branch1x1 = conv2d(x, 32, 1, name=f'{name}.1x1')

    branch3x3 = conv2d(x, 32, 1, name=f"{name}.middle_1x1")
    branch3x3 = conv2d(branch3x3, 32, 3, padding="same", name=f"{name}.middle_3x3")

    branch3x3_dbl = conv2d(x, 32, 1, name=f"{name}.dbl_1x1")
    branch3x3_dbl = conv2d(branch3x3_dbl, 32, 3, padding="same", name=f"{name}.dbl_3x3_1")
    branch3x3_dbl = conv2d(branch3x3_dbl, 32, 3, padding="same", name=f"{name}.dbl_3x3_2")

    branches = layers.Concatenate(name=f"{name}.concat")([branch1x1, branch3x3, branch3x3_dbl])

    up = conv2d(branches, x.shape[-1], 1, name=f"{name}.up")

    x = layers.Add(name=f"{name}.add")([x * scale, up])
    return x


def inception_res_b(x, scale, name):
    branch1x1 = conv2d(x, 128, 1, name=f'{name}.1x1')

    branch7x7 = conv2d(x, 128, 1, name=f"{name}.middle_1x1")
    branch7x7 = conv2d(branch7x7, 128, (1, 7), padding="same", name=f"{name}.1x7")
    branch7x7 = conv2d(branch7x7, 128, (7, 1), padding="same", name=f"{name}.7x1")

    branches = layers.Concatenate(name=f"{name}.concat")([branch1x1, branch7x7])

    up = conv2d(branches, x.shape[-1], 1, name=f"{name}.up")

    x = layers.Add(name=f"{name}.add")([x * scale, up])
    return x


def inception_res_c(x, scale, name):
    branch1x1 = conv2d(x, 192, 1, name=f'{name}.1x1')

    branch3x3 = conv2d(x, 192, 1, name=f"{name}.middle_1x1")
    branch3x3 = conv2d(branch3x3, 192, (1, 3), padding="same", name=f"{name}.1x3")
    branch3x3 = conv2d(branch3x3, 192, (3, 1), padding="same", name=f"{name}.3x1")

    branches = layers.Concatenate(name=f"{name}.concat")([branch1x1, branch3x3])

    up = conv2d(branches, x.shape[-1], 1, name=f"{name}.up")

    x = layers.Add(name=f"{name}.add")([x * scale, up])
    return x


def reduce_a(x, name):
    branch3x3 = conv2d(x, 384, 3, strides=2, name=f"{name}.3x3")

    branch3x3_dbl = conv2d(x, 256, 1, name=f"{name}.1x1")
    branch3x3_dbl = conv2d(branch3x3_dbl, 256, 3, padding="same", name=f"{name}.dbl_3x3_1")
    branch3x3_dbl = conv2d(branch3x3_dbl, 384, 3, strides=2, name=f"{name}.dbl_3x3_2")

    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid")(x)

    branches = layers.Concatenate(name=f"{name}.concat")([branch3x3, branch3x3_dbl, branch_pool])

    return branches


def reduce_b(x, name):
    branch_1 = conv2d(x, 256, 1, name=f"{name}.branch1_1x1")
    branch_1 = conv2d(branch_1, 384, 3, strides=2, name=f"{name}.branch1_3x3")

    branch_2 = conv2d(x, 256, 1, name=f"{name}.branch2_1x1")
    branch_2 = conv2d(branch_2, 256, 3, strides=2, name=f"{name}.branch2_3x3")

    branch_3 = conv2d(x, 256, 1, name=f"{name}.branch3_1x1")
    branch_3 = conv2d(branch_3, 256, 3, padding="same", name=f"{name}.branch3_3x3_1")
    branch_3 = conv2d(branch_3, 256, 3, strides=2, name=f"{name}.branch3_3x3_2")

    branch_pool = layers.MaxPooling2D(3, strides=2, padding="valid", name=f"{name}.branch_pool")(x)
    branches = layers.Concatenate(name=f"{name}.concat")([branch_1, branch_2, branch_3, branch_pool])

    return branches


def stem(x):
    x = conv2d(x, 32, 3, strides=2, name="conv_1")
    x = conv2d(x, 32, 3, padding="same", name="conv_2")
    x = conv2d(x, 64, 3, name="conv_3")
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d(x, 80, 1, name="conv_4")
    x = conv2d(x, 192, 3, name="conv_5")
    x = conv2d(x, 256, 3, strides=2, name="conv_6")

    return x


def InceptionResNetV1(num_cls, scale=0.3):
    
    _input = layers.Input(shape=(299, 299, 3), name="Input")

    x = stem(_input)

    for i in range(5):
        x = inception_res_a(x, scale, name=f"inceptionRes_a{i}")

    x = reduce_a(x, name="reduce_a")

    for i in range(10):
        x = inception_res_b(x, scale, name=f"inceptionRes_b{i}")

    x = reduce_b(x, name="reduce_b")

    for i in range(5):
        x = inception_res_c(x, scale, name=f"inceptionRes_c{i}")

    x = layers.GlobalAvgPool2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)

    return keras.Model(inputs=_input, outputs=x, name="InceptionResNetV1")
