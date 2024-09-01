from keras import models, layers, utils


def depthwise_separable_conv_func(x, _filters, alpha, strides=2, block_id=1):
    _filters = int(_filters * alpha)
    name = f"block{block_id}"
    return models.Sequential([
        layers.DepthwiseConv2D(kernel_size=3, strides=strides,
                               padding="same", name=f"{name}.depthwise"),
        layers.BatchNormalization(name=f"{name}.bn_1"),
        layers.ReLU(6.0, name=f"{name}.act_1"),
        layers.Conv2D(filters=_filters, kernel_size=1, strides=1, name=f"{name}.pointwise"),
        layers.BatchNormalization(name=f"{name}.bn_2"),
        layers.ReLU(6.0, name=f"{name}.act_2"),
    ],
        name=name)(x)


def MobileNetV1(num_cls, alpha=1.0):
    _input = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", name="first_conv")(_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = depthwise_separable_conv_func(x, 64, alpha, block_id=1)

    x = depthwise_separable_conv_func(
        x, 128, alpha, strides=2, block_id=2
    )
    x = depthwise_separable_conv_func(x, 128, alpha, block_id=3)

    x = depthwise_separable_conv_func(
        x, 256, alpha, strides=2, block_id=4
    )
    x = depthwise_separable_conv_func(x, 256, alpha, block_id=5)

    x = depthwise_separable_conv_func(
        x, 512, alpha, strides=2, block_id=6
    )
    x = depthwise_separable_conv_func(x, 512, alpha, block_id=7)
    x = depthwise_separable_conv_func(x, 512, alpha, block_id=8)
    x = depthwise_separable_conv_func(x, 512, alpha, block_id=9)
    x = depthwise_separable_conv_func(x, 512, alpha, block_id=10)
    x = depthwise_separable_conv_func(x, 512, alpha, block_id=11)

    x = depthwise_separable_conv_func(
        x, 1024, alpha, strides=2, block_id=12
    )
    x = depthwise_separable_conv_func(x, 1024, alpha, block_id=13)

    x = layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Conv2D(num_cls, (1, 1), padding="same", name="conv_preds")(x)
    x = layers.Reshape((num_cls,), name="reshape_2")(x)
    x = layers.Activation(activation='softmax', name="output")(x)

    return models.Model(inputs=_input, outputs=x, name=f"MobileNetV1")


if __name__ == "__main__":
    model = MobileNetV1(1000, 1)

