from keras import layers
import keras


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_res_block(x, _filter, strides, alpha, expansion_factor, block_id):
    in_channels = x.shape[-1]
    _filter = make_divisible(int(_filter * alpha), 8)
    name = f"block_{block_id}_"

    # Expand conv
    output = layers.Conv2D(
        in_channels * expansion_factor,
        1,
        1,
        name=f"{name}.expand",
        padding="same",
    )(x)
    output = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=f"{name}.expand.bn")(output)
    output = layers.ReLU(max_value=6, name=f"{name}.expand.relu")(output)

    output = layers.DepthwiseConv2D(
        3, strides=strides, name=f"{name}.depthwise", padding="same"
    )(output)
    output = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=f"{name}.depthwise.bn")(output)
    output = layers.ReLU(max_value=6, name=f"{name}.depthwise.relu")(output)

    output = layers.Conv2D(_filter, 1, 1, name=f"{name}.project")(output)
    output = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=f"{name}.project.bn")(output)

    if strides == 1 and x.shape[-1] == output.shape[-1]:
        output = layers.Add(name=f"{name}.add")([x, output])
    return output


def MobileNetV2(num_cls, alpha=1):
    _input = layers.Input(shape=(224, 224, 3), name="Input")

    _first_filter = make_divisible(32 * alpha, 8)
    x = layers.Conv2D(_first_filter, kernel_size=3, strides=2, padding="same", name="conv1")(_input)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='conv1.bn')(x)
    x = layers.ReLU(6., name='conv1.relu6')(x)

    x = inverted_res_block(x, _filter=16, strides=1, expansion_factor=1, block_id=1, alpha=alpha)

    x = inverted_res_block(x, _filter=24, strides=2, expansion_factor=6, block_id=2, alpha=alpha)
    x = inverted_res_block(x, _filter=24, strides=1, expansion_factor=6, block_id=3, alpha=alpha)

    x = inverted_res_block(x, _filter=32, strides=2, expansion_factor=6, block_id=4, alpha=alpha)
    x = inverted_res_block(x, _filter=32, strides=1, expansion_factor=6, block_id=5, alpha=alpha)
    x = inverted_res_block(x, _filter=32, strides=1, expansion_factor=6, block_id=6, alpha=alpha)

    x = inverted_res_block(x, _filter=64, strides=2, expansion_factor=6, block_id=7, alpha=alpha)
    x = inverted_res_block(x, _filter=64, strides=1, expansion_factor=6, block_id=8, alpha=alpha)
    x = inverted_res_block(x, _filter=64, strides=1, expansion_factor=6, block_id=9, alpha=alpha)
    x = inverted_res_block(x, _filter=64, strides=1, expansion_factor=6, block_id=10, alpha=alpha)

    x = inverted_res_block(x, _filter=96, strides=2, expansion_factor=6, block_id=11, alpha=alpha)
    x = inverted_res_block(x, _filter=96, strides=1, expansion_factor=6, block_id=12, alpha=alpha)
    x = inverted_res_block(x, _filter=96, strides=1, expansion_factor=6, block_id=13, alpha=alpha)

    x = inverted_res_block(x, _filter=160, strides=1, expansion_factor=6, block_id=14, alpha=alpha)
    x = inverted_res_block(x, _filter=160, strides=1, expansion_factor=6, block_id=15, alpha=alpha)
    x = inverted_res_block(x, _filter=160, strides=1, expansion_factor=6, block_id=16, alpha=alpha)

    x = inverted_res_block(x, _filter=320, strides=1, expansion_factor=6, block_id=17, alpha=alpha)

    last_filter = make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280

    x = layers.Conv2D(last_filter, kernel_size=1, name="conv2")(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name="conv2.bn")(x)
    x = layers.ReLU(6.0, name="conv2.relu6")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)

    return keras.Model(inputs=_input, outputs=x, name=f"MobileNetV2")


if __name__ == "__main__":
    model = MobileNetV2(1000, 1)
