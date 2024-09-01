import keras
from keras import layers


def conv_bn(x, _filter, kernel_size, strides, padding="valid", name=None):
    x = layers.Conv2D(_filter, kernel_size, strides, padding=padding, name=f"{name}.conv")(x)
    x = layers.BatchNormalization(name=f"{name}.bn")(x)
    return x


def separable_bn(x, _filter, kernel_size, strides, padding="valid", name=None):
    x = layers.SeparableConv2D(_filter, kernel_size, strides, padding=padding, name=f"{name}.spconv")(x)
    x = layers.BatchNormalization(name=f"{name}.sp.bn")(x)
    return x


def entry_flow_block(x, _filter, name):
    x = conv_bn(x, _filter, 3, 1, padding="same", name=name)
    x = layers.ReLU(name=f"{name}.relu")(x)
    x = separable_bn(x, _filter, 3, 1, padding="same", name=name)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    return x


def entry_flow(x):
    x = conv_bn(x, 32, 3, 2, padding='same', name='ef_1')
    x = layers.ReLU(name='ef_1.relu')(x)
    x = conv_bn(x, 64, 3, 1, padding='same', name='ef_2')
    x = layers.ReLU(name='ef_2.relu')(x)

    branch = x
    branch = conv_bn(branch, 128, 1, strides=2, padding="same", name="ef.branch_1")
    x = entry_flow_block(x, 128, name="ef.block_1")
    x = layers.Add(name="ef.add_1")([x, branch])

    branch = x
    branch = conv_bn(branch, 256, 1, strides=2, padding="same", name="ef.branch_2")
    x = entry_flow_block(x, 256, name="ef.block_2")
    x = layers.Add(name="ef.add_2")([x, branch])

    branch = x
    branch = conv_bn(branch, 728, 1, strides=2, padding="same", name="ef.branch_3")
    x = entry_flow_block(x, 728, name="ef.block_3")
    x = layers.Add(name="ef.add_3")([x, branch])

    return x


def middle_flow(inputs, name):
    x = layers.ReLU(name=f"{name}.relu1")(inputs)
    x = separable_bn(x, 728, 3, 1, padding="same", name=f"{name}_1")

    x = layers.ReLU(name=f"{name}.relu2")(x)
    x = separable_bn(x, 728, 3, 1, padding="same", name=f"{name}_2")

    x = layers.ReLU(name=f"{name}.relu3")(x)
    x = separable_bn(x, 728, 3, 1, padding="same", name=f"{name}_3")

    return layers.Add(name=f"{name}.add")([inputs, x])


def exit_flow(inputs):
    x = layers.ReLU(name="exitflow.relu1")(inputs)
    x = separable_bn(x, 728, 3, 1, padding="same", name="exitflow_1")

    x = layers.ReLU(name="exitflow.relu2")(x)
    x = separable_bn(x, 1024, 3, 1, padding="same", name="exitflow_2")

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    branch = conv_bn(inputs, 1024, 1, strides=2, padding="same")

    x = layers.Add(name="exitflow.add")([x, branch])

    x = separable_bn(x, 1536, 3, 1, padding="same", name="exitflow_3")
    x = layers.ReLU(name="exitflow.relu3")(x)

    x = separable_bn(x, 2048, 3, 1, padding="same", name="exitflow_4")
    x = layers.ReLU(name="exitflow.relu4")(x)

    x = layers.GlobalAvgPool2D(name="avg_pool")(x)

    return x


def Xception(num_cls):
    _input = layers.Input(shape=(299, 299, 3), name="input")
    x = entry_flow(_input)

    for i in range(8):
        x = middle_flow(x, name=f"md.block_{i + 1}")
    x = exit_flow(x)

    x = layers.Dense(num_cls, activation="softmax", name="output")(x)
    return keras.Model(inputs=_input, outputs=x, name="XceptionNet")


