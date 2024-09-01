from typing import Callable
import tensorflow as tf
import keras
from keras import layers


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Activation("hard_swish")(x)


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def se_block(x, _filter, se_ratio, name):
    output = layers.GlobalAveragePooling2D(keepdims=True, name=f'{name}.avg_pool')(x)

    expansion_factor = make_divisible(int(_filter * se_ratio), 8)

    output = layers.Conv2D(expansion_factor, 1, name=f'{name}.conv_1')(output)

    output = layers.ReLU(name=f"{name}.relu")(output)

    output = layers.Conv2D(_filter, 1, name=f'{name}.conv_2')(output)

    output = hard_sigmoid(output)

    x = layers.Multiply(name=f"{name}.mul")([x, output])

    return x


def inverted_res_block(x, expansion, _filter, kernel_size, strides, se_ratio, activation, block_id):
    _input = x

    expansion = make_divisible(_filter * expansion, 8)

    name = f"block_{block_id}"

    x = layers.Conv2D(expansion, 1, padding="same", name=f"{name}.expand")(x)
    x = layers.BatchNormalization(name=f"{name}.expand.bn")(x)
    x = activation(x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                               padding="same",
                               name=f"{name}.depthwise")(x)
    x = layers.BatchNormalization(name=f"{name}.depthwise.bn")(x)
    x = activation(x)

    if se_ratio:
        x = se_block(x, expansion, se_ratio, name=f"{name}.se")

    x = layers.Conv2D(_filter, 1, padding='same', name=f"{name}.project")(x)
    x = layers.BatchNormalization(name=f"{name}.project.bn")(x)

    if strides == 1 and _input.shape[-1] == x.shape[-1]:
        x = layers.Add(name=f"{name}.add")([_input, x])

    return x


def MobileNet(num_cls, alpha, se_ratio, arch_builder, name) -> keras.Model:
    _input = keras.Input(shape=(224, 224, 3), name="input")

    x = layers.Conv2D(32, 3, strides=2, padding="same", name="conv_1")(_input)
    x = layers.BatchNormalization(name="conv_1.bn")(x)
    x = hard_swish(x)

    x = arch_builder(x, se_ratio)

    x = layers.AveragePooling2D(pool_size=7, name="avg_pool")(x)
    x = layers.Conv2D(1024, 1, name="conv_2")(x)
    x = hard_swish(x)

    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(num_cls, 1, name="output")(x)

    x = layers.Flatten()(x)
    x = layers.Softmax()(x)

    return keras.Model(inputs=_input, outputs=x, name=name)


def MobileNetV3Large(num_cls: int, alpha: float = 1.0, se_ratio=0.25) -> keras.Model:
    def arch_builder(x, se_ratio):
        def depth(d: float) -> int:
            return make_divisible(int(d * alpha))

        x = inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)

        x = inverted_res_block(x, 3, depth(40), 5, 2, se_ratio, relu, 3)
        x = inverted_res_block(x, 3, depth(40), 5, 1, se_ratio, relu, 4)
        x = inverted_res_block(x, 3, depth(40), 5, 1, se_ratio, relu, 5)

        x = inverted_res_block(x, 6, depth(80), 3, 2, None, hard_swish, 6)
        x = inverted_res_block(x, 2.5, depth(80), 3, 1, None, hard_swish, 7)
        x = inverted_res_block(x, 2.3, depth(80), 3, 1, None, hard_swish, 8)
        x = inverted_res_block(x, 2.3, depth(80), 3, 1, None, hard_swish, 9)

        x = inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, hard_swish, 10)
        x = inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, hard_swish, 11)
        x = inverted_res_block(x, 6, depth(160), 5, 2, se_ratio, hard_swish, 12)
        x = inverted_res_block(x, 6, depth(160), 5, 1, se_ratio, hard_swish, 13)
        x = inverted_res_block(x, 6, depth(160), 5, 1, se_ratio, hard_swish, 14)

        return x

    return MobileNet(num_cls, alpha, se_ratio, arch_builder, f"MobileNetV3Large")


def MobileNetV3Small(num_cls: int, alpha: float = 1.0, se_ratio=0.25) -> keras.Model:
    def arch_builder(x, se_ratio):
        def depth(d: float) -> int:
            return make_divisible(int(d * alpha))

        x = inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)  # Block 0
        x = inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)  # Block 1
        x = inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)  # Block 2
        x = inverted_res_block(x, 4, depth(40), 5, 2, se_ratio, hard_swish, 3)  # Block 3
        x = inverted_res_block(x, 6, depth(40), 5, 1, se_ratio, hard_swish, 4)  # Block 4
        x = inverted_res_block(x, 6, depth(40), 5, 1, se_ratio, hard_swish, 5)  # Block 5
        x = inverted_res_block(x, 3, depth(48), 5, 1, se_ratio, hard_swish, 6)  # Block 6
        x = inverted_res_block(x, 3, depth(48), 5, 1, se_ratio, hard_swish, 7)  # Block 7
        x = inverted_res_block(x, 6, depth(96), 5, 2, se_ratio, hard_swish, 8)  # Block 8
        x = inverted_res_block(x, 6, depth(96), 5, 1, se_ratio, hard_swish, 9)  # Block 9
        x = inverted_res_block(x, 6, depth(96), 5, 1, se_ratio, hard_swish, 10)  # Block 10
        return x

    return MobileNet(num_cls, alpha, se_ratio, arch_builder, f"MobileNetV3Small")
