import copy
import math

from keras.src import layers
import keras

DEFAULT_BLOCKS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "strides": 1,
        "se_ratio": 0.25,
    },
]
IMAGENET_STDDEV_RGB = [0.229, 0.224, 0.225]


def round_filters(filters, width_coefficient, divisor=8):
    
    filters *= width_coefficient
    new_filters = max(
        divisor, int(filters + divisor / 2) // divisor * divisor
    )
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    
    return int(math.ceil(depth_coefficient * repeats))


def se_block(x, _filter, expand_ratio, se_ratio, activation, name):
    filters_se = max(1, int(_filter * se_ratio))
    se = layers.GlobalAveragePooling2D(name=f"{name}.se.squeeze")(x)

    se = layers.Reshape((1, 1, _filter*expand_ratio), name=f"{name}.se.reshape")(se)
    se = layers.Conv2D(
        filters_se,
        1,
        padding="same",
        activation=activation,
        name=f"{name}.se.reduce",
    )(se)
    se = layers.Conv2D(
        _filter*expand_ratio,
        1,
        padding="same",
        activation="sigmoid",
        name=f"{name}.se.expand",
    )(se)
    return layers.Multiply(name=f"{name}.se.excite")([x, se])


def mobile_v3_block(inputs, activation="swish", drop_rate=0.0, name="", filters_in=32,
                    filters_out=16, kernel_size=3, strides=1,
                    expand_ratio=1, se_ratio=0.0):
    


    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            name=f"{name}.expand.conv",
        )(inputs)
        x = layers.BatchNormalization(name=f"{name}.expand.bn")(x)
        x = layers.Activation(activation, name=f"{name}.expand.{activation}")(x)
    else:
        x = inputs

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}.dw.conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}.bn")(x)
    x = layers.Activation(activation, name=f"{name}.{activation}")(x)

    if se_ratio is not None:
        x = se_block(x, filters_in, expand_ratio, se_ratio, activation, name)

    x = layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        name=f"{name}.project.conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}.project.bn")(x)
    if strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=f"{name}.drop"
            )(x)
        x = layers.Add(name=f"{name}.add")([x, inputs])
    return x


def EfficientNet(
        num_cls,
        input_shape,
        width_coefficient,
        depth_coefficient,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation="swish",
        blocks_args="default",
        model_name="efficientnet",
):
    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS

    _input = layers.Input(shape=input_shape, name="input")

    x = layers.Conv2D(
        round_filters(32, width_coefficient, depth_divisor),
        3,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem.conv",
    )(_input)
    x = layers.BatchNormalization(name="stem.bn")(x)
    x = layers.Activation(activation, name=f"stem.{activation}")(x)

    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args["repeats"], depth_coefficient) for args in blocks_args))
    for i, args in enumerate(blocks_args):
        assert args["repeats"] > 0
        args["filters_in"] = round_filters(args["filters_in"], width_coefficient, depth_divisor)
        args["filters_out"] = round_filters(args["filters_out"], width_coefficient, depth_divisor)

        for j in range(round_repeats(args.pop("repeats"), depth_coefficient)):

            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            x = mobile_v3_block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                name=f"block{i + 1}_{chr(j + 97)}",
                **args,
            )
            b += 1

    x = layers.Conv2D(
        round_filters(1280, width_coefficient, depth_divisor),
        1,
        padding="same",
        use_bias=False,
        name="top.conv",
    )(x)
    x = layers.BatchNormalization(name="top.bn")(x)
    x = layers.Activation(activation, name="top.activation")(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="top.dropout")(x)

    x = layers.Dense(num_cls, activation='softmax', name="outputs")(x)

    model = keras.Model(inputs=_input, outputs=x, name=model_name)

    return model
