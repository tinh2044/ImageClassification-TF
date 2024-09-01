import keras
from keras import layers


def conv_block(_filter, kernel_size, strides, padding="valid", act=None, name=None):
    conv = keras.Sequential([

        layers.Conv2D(_filter, kernel_size=kernel_size, strides=strides, padding=padding),
        layers.BatchNormalization(),

    ], name=name)
    if act is not None:
        conv.add(layers.Activation(act))

    return conv


def res_block_v1(x, _filter, kernel_size, strides, use_shortcut=None, name=None):
    if use_shortcut:
        shortcut = conv_block(4 * _filter, kernel_size=1, strides=strides, padding="valid", name=f"{name}.shortcut")(x)

    else:
        shortcut = x

    x = conv_block(_filter, 1, strides=strides, padding="valid", act='relu', name=f"{name}.conv_1")(x)
    x = conv_block(_filter, kernel_size=kernel_size, strides=1, padding="same", act='relu', name=f"{name}.conv_2")(x)
    x = conv_block(4 * _filter, kernel_size=1, strides=1, act='relu', name=f"{name}.conv_3")(x)

    x = layers.Add(name=f"{name}.add")([shortcut, x])
    x = layers.ReLU(name=f"{name}.relu")(x)

    return x


def stack_residual_v1(x, _filter, num_block, strides, name):
    x = res_block_v1(x, _filter, kernel_size=3, strides=strides, use_shortcut=True, name=f"{name}.block_1")

    for i in range(2, num_block + 1):
        x = res_block_v1(x, _filter, kernel_size=3, strides=1, use_shortcut=None, name=f"{name}.block_{i}")

    return x


def res_block_v2(x, _filter, strides, use_shortcut, name):
    x = layers.BatchNormalization(name=f"{name}.preact_bn")(x)
    x = layers.ReLU(name=f"{name}.preact_relu")(x)

    if use_shortcut:
        shortcut = conv_block(4 * _filter, kernel_size=1, strides=strides, padding="valid", name=f"{name}.shortcut")(x)

    else:
        shortcut = x

    x = conv_block(_filter, 1, strides=strides, padding="valid", act='relu', name=f"{name}.conv_1")(x)
    x = conv_block(_filter, kernel_size=3, strides=1, padding="same", act='relu', name=f"{name}.conv_2")(x)
    x = conv_block(4 * _filter, kernel_size=1, strides=1, act=None, name=f"{name}.conv_3")(x)

    x = layers.Add(name=f"{name}.add")([shortcut, x])

    return x


def stack_residual_v2(x, filters, blocks, strides=2, name=None):
    x = res_block_v2(x, filters, 1, use_shortcut=True, name=f"{name}.block1")
    for i in range(2, blocks):
        x = res_block_v2(x, filters, 1, use_shortcut=False, name=f"{name}.block" + str(i))
    x = res_block_v2(
        x, filters, strides=strides, use_shortcut=False, name=f"{name}.block" + str(blocks)
    )
    return x


def res_net(
        num_cls,
        arch_builder,
        preact,
        use_bias,
        model_name="resnet",
):
    _input = layers.Input(shape=(224, 224, 3), name="Input")

    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=use_bias, name="Conv1")(_input)

    if not preact:
        x = layers.BatchNormalization(name="Conv1_bn")(x)
        x = layers.ReLU(name="Con1_relu")(x)

    x = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(x)

    x = arch_builder(x)

    if preact:
        x = layers.BatchNormalization(name="post_bn")(x)
        x = layers.ReLU(name="post_relu")(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation='softmax', name="output")(x)

    return keras.Model(inputs=_input, outputs=x, name=model_name)
