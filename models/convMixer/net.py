import tensorflow as tf
from keras import layers
import keras


def act_bn(x, name=None):
    x = layers.Activation("gelu", name=f"{name}.gelu")(x)
    return layers.BatchNormalization(name=f"{name}.bn")(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size, name="stemc.conv")(x)
    return act_bn(x, name="stem")


def conv_mixer_block(inputs, _filter: int, kernel_size: int, name=None):
    x = inputs
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", name=f"{name}.dwconv")(x)
    x = act_bn(x, name=f"{name}.dw")
    x = layers.Add(name=f"{name}.add")([x, inputs])

    # Pointwise convolution.
    x = layers.Conv2D(_filter, kernel_size=1, name=f"{name}.pwconv")(x)
    x = act_bn(x, name=f"{name}.pw")

    return x


def ConvMixer(num_cls, _filter=256, depth=8, kernel_size=5, patch_size=2, name="ConvMixer"):
    _input = keras.Input((224, 224, 3), name="_input")

    x = conv_stem(_input, _filter, patch_size)

    for i in range(depth):
        x = conv_mixer_block(x, _filter, kernel_size, name=f"block{i + 1}")

    x = layers.GlobalAvgPool2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)

    return keras.Model(inputs=_input, outputs=x, name=name)
