from keras import layers
import keras


def separable_conv_block(x, filters, kernel_size=3, strides=1, block_id=None):
    x = layers.ReLU()(x)
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(x, (kernel_size, kernel_size)),
            name=f"block{block_id}.separable_1.pad",
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        name=f"block{block_id}.separable_1",
        padding=conv_pad,
        use_bias=False,
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9997,
        epsilon=1e-3,
        name=f"block{block_id}.separable_1.bn",
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        name=f"block{block_id}.separable_2",
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9997,
        epsilon=1e-3,
        name=f"block{block_id}.separable_2.bn",
    )(x)
    return x


def adjust_block(p, ip, filters, block_id=None):
    if p is None:
        p = ip
    elif p.shape[-2] != ip.shape[-2]:

        p = layers.Activation("relu", name=f"block{block_id}.adjust.relu_1")(p)
        p1 = layers.AveragePooling2D(pool_size=1,
                                     strides=2,
                                     padding="valid",
                                     name=f"block{block_id}.adjust.avg_pool_1"
                                     )(p)
        p1 = layers.Conv2D(
            filters // 2,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=f"block{block_id}.adjust.conv_1",
            kernel_initializer="he_normal",
        )(p1)
        p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
        p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
        p2 = layers.AveragePooling2D(
            pool_size=1,
            strides=2,
            padding="valid",
            name=f"block{block_id}.adjust.avg_pool_2",
        )(p2)
        p2 = layers.Conv2D(
            filters // 2,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=f"block{block_id}.adjust.conv_2",
            kernel_initializer="he_normal",
        )(p2)
        p = layers.concatenate([p1, p2])
        p = layers.BatchNormalization(
            momentum=0.9997,
            epsilon=1e-3,
            name=f"block{block_id}.adjust.bn",
        )(p)
    elif p.shape[-1] != filters:

        p = layers.ReLU(name=f"block{block_id}.adjust.relu")(p)
        p = layers.Conv2D(
            filters,
            1,
            strides=1,
            padding="same",
            name=f"block{block_id}.adjust.projection",
            use_bias=False,
            kernel_initializer="he_normal",
        )(p)
        p = layers.BatchNormalization(
            momentum=0.9997,
            epsilon=1e-3,
            name=f"block{block_id}.adjust.projection.bn",
        )(p)
    return p


def normal_a_cell(inputs, p, filters, block_id=None):
    p = adjust_block(p, inputs, filters, block_id)
    h = layers.ReLU()(inputs)
    h = layers.Conv2D(
        filters,
        1,
        strides=1,
        padding="same",
        name=f"block{block_id}.normal.conv_1",
        use_bias=False,
        kernel_initializer="he_normal")(h)

    h = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name=f"block{block_id}.normal.bn_1")(h)

    x1_1 = separable_conv_block(h, filters, kernel_size=5, block_id=f"block{block_id}.normal.left1")
    x1_2 = separable_conv_block(p, filters, block_id=f"block{block_id}.normal.right1")
    x1 = layers.add([x1_1, x1_2], name=f"block{block_id}.normal.add_1")

    x2_1 = separable_conv_block(p, filters, 5, block_id=f"block{block_id}.normal.left2")
    x2_2 = separable_conv_block(p, filters, 3, block_id=f"block{block_id}.normal.right2")

    x2 = layers.Add(name=f"block{block_id}.normal.add_2")([x2_1, x2_2])

    x3 = layers.AveragePooling2D(3, strides=1, padding="same", name=f"block{block_id}.normal.left3")(h)
    x3 = layers.Add(name=f"block{block_id}.normal.add_3")([x3, p])

    x4_1 = layers.AveragePooling2D(3, strides=1, padding="same", name=f"block{block_id}.normal.left4")(p)
    x4_2 = layers.AveragePooling2D(3, strides=1, padding="same", name=f"block{block_id}.normal.right4")(p)

    x4 = layers.Add(name=f"block{block_id}.normal.add_4")([x4_1, x4_2])

    x5 = separable_conv_block(h, filters, block_id=f"block{block_id}.normal.left5")

    x5 = layers.Add(name=f"block{block_id}.normal.add_5")([x5, h])

    x = layers.Concatenate(name=f"block{block_id}.normal.concat")([p, x1, x2, x3, x4, x5])
    return x, inputs


def reduction_a_cell(inputs, p, filters, block_id=None):
    p = adjust_block(p, inputs, filters, block_id)
    h = layers.Activation("relu")(inputs)
    h = layers.Conv2D(
        filters,
        1,
        strides=1,
        padding="same",
        name=f"block{block_id}.reduction.conv_1",
        use_bias=False,
        kernel_initializer="he_normal",
    )(h)
    h = layers.BatchNormalization(
        momentum=0.9997,
        epsilon=1e-3,
        name=f"block{block_id}.reduction.bn_1",
    )(h)
    h3 = layers.ZeroPadding2D(
        padding=correct_pad(h, (3, 3)),
        name=f"block{block_id}.reduction.pad_1",
    )(h)

    x1_1 = separable_conv_block(
        h,
        filters,
        5,
        strides=2,
        block_id=f"block{block_id}.reduction.left1",
    )
    x1_2 = separable_conv_block(
        p,
        filters,
        7,
        strides=2,
        block_id=f"block{block_id}.reduction.right1",
    )
    x1 = layers.Add(name=f"block{block_id}.reduction.add_1")([x1_1, x1_2])

    x2_1 = layers.MaxPooling2D(
        3,
        strides=2,
        padding="valid",
        name=f"block{block_id}.reduction.left2",
    )(h3)
    x2_2 = separable_conv_block(
        p,
        filters,
        7,
        strides=2,
        block_id=f"block{block_id}.reduction.right2",
    )
    x2 = layers.Add(name=f"block{block_id}.reduction.add_2")([x2_1, x2_2])

    x3_1 = layers.AveragePooling2D(
        3,
        strides=2,
        padding="valid",
        name=f"block{block_id}.reduction.left3",
    )(h3)
    x3_2 = separable_conv_block(
        p,
        filters,
        5,
        strides=2,
        block_id=f"block{block_id}.reduction.right3",
    )
    x3 = layers.Add(name=f"block{block_id}.reduction.add3")([x3_1, x3_2])

    x4 = layers.AveragePooling2D(
        3,
        strides=1,
        padding="same",
        name=f"block{block_id}.reduction.left4",
    )(x1)
    x4 = layers.Add()([x2, x4])

    x5_1 = separable_conv_block(
        x1, filters, 3, block_id=f"block{block_id}.reduction.left4"
    )
    x5_2 = layers.MaxPooling2D(
        3,
        strides=2,
        padding="valid",
        name=f"block{block_id}.reduction.right5",
    )(h3)
    x5 = layers.Add(name=f"block{block_id}.reduction.add4")([x5_1, x5_2])

    x = layers.Concatenate(name=f"block{block_id}.reduction.concat")([x2, x3, x4, x5])
    return x, inputs


def correct_pad(inputs, kernel_size):
    img_dim = 1
    input_size = inputs.shape[img_dim: (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = 1
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def nas_net(
        num_cls,
        name,
        penultimate_filters=4032,
        num_blocks=6,
        stem_block_filters=96,
        skip_reduction=True,
        filter_multiplier=2,
):
    _input = layers.Input(shape=(224, 224, 3), name="input")

    filters = penultimate_filters // 24

    x = layers.Conv2D(
        stem_block_filters,
        3,
        strides=5,
        padding="valid",
        use_bias=False,
        name="stem.conv1",
        kernel_initializer="he_normal",
    )(_input)

    x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name="stem.bn1")(x)

    p = None

    x, p = reduction_a_cell(x, p, filters // (filter_multiplier ** 2), block_id="stem_1")
    x, p = reduction_a_cell(x, p, filters // filter_multiplier, block_id="stem_2")

    for i in range(num_blocks):
        x, p = normal_a_cell(x, p, filters, block_id=f"{i}")

    x, p0 = reduction_a_cell(x, p, filters * filter_multiplier, block_id=f"{num_blocks}")

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = normal_a_cell(x, p,
                             filters * filter_multiplier,
                             block_id=f"{num_blocks + i + 1}")

    x, p0 = reduction_a_cell(x, p,
                             filters * filter_multiplier ** 2,
                             block_id="reduce_%d" % (2 * num_blocks))

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = normal_a_cell(x, p,
                             filters * filter_multiplier ** 2,
                             block_id="%d" % (2 * num_blocks + i + 1))

    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)

    model = keras.Model(inputs=_input, outputs=x, name=name)

    return model
