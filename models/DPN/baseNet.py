import keras
from keras import layers


def conv_bn(x, _filter, kernel_size, padding="same", strides=1, groups=1, name=None):
    x = layers.Conv2D(_filter, kernel_size, strides=strides, padding=padding, groups=groups, name=f"{name}.conv")(x)
    x = layers.BatchNormalization(name=f"{name}.bn")(x)
    x = layers.ReLU(name=f"{name}.relu")(x)
    return x


def dual_path_block(x, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups=1, block_type='normal', b=False, name=None):
    if isinstance(x, tuple):
        x_in = layers.Concatenate(name=f"{name}.concat_r_d")(x)
    else:
        x_in = x
    key_stride = 2 if block_type == 'down' else 1
    has_proj = block_type in ['proj', 'down']

    if has_proj and key_stride == 2:
        x_s = conv_bn(x_in, _filter=num_1x1_c + 2 * inc, kernel_size=2, strides=key_stride, name=f"{name}.1x1_s2")
    else:
        x_s = conv_bn(x_in, _filter=num_1x1_c + 2 * inc, kernel_size=1, strides=key_stride, name=f"{name}.1x1_s1")

    if has_proj:
        x_s1 = x_s[:, :, :, :num_1x1_c]
        x_s2 = x_s[:, :, :, num_1x1_c:]
    else:
        x_s1, x_s2 = x

    x_in = conv_bn(x_in, _filter=num_1x1_a, kernel_size=1, name=f"{name}.1x1_a")
    x_in = conv_bn(x_in, _filter=num_3x3_b, kernel_size=3, strides=key_stride, groups=groups, name=f"{name}.3x3_b")

    if b:
        x_in = layers.BatchNormalization()(x_in)
        x_in = layers.ReLU()(x_in)
        out1 = layers.Conv2D(num_1x1_c, kernel_size=1, use_bias=False, name=f"{name}.1x1_c1")(x_in)
        out2 = layers.Conv2D(inc, kernel_size=1, use_bias=False, name=f"{name}.1x1_c2")(x_in)
    else:
        x_in = conv_bn(x_in, _filter=num_1x1_c + inc, kernel_size=1, name=f"{name}.1x1_c")
        out1 = x_in[:, :, :, :num_1x1_c]
        out2 = x_in[:, :, :, num_1x1_c:]

    resid = layers.Add(name=f"{name}.add_res")([x_s1, out1])
    dense = layers.Concatenate(name=f"{name}.concat_dense")([x_s2, out2])

    return resid, dense


def dpn(num_cls, small=False, num_init_features=64, k_r=96, groups=32,
        b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        test_time_pool=False, name=None):
    _input = layers.Input(shape=(224, 224, 3), name="input")
    x = conv_bn(_input, num_init_features, 7, padding="same", strides=2, name="conv1")
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="pool1")(x)

    # conv2
    bw = 256
    _filter = (k_r * bw) // 256
    x = dual_path_block(x, _filter, _filter, bw, inc_sec[0], groups, 'proj', b, name="conv2_1")
    for i in range(2, k_sec[0] + 1):
        x = dual_path_block(x, _filter, _filter, bw, inc_sec[0], groups, 'normal', b, name=f"conv2_{i}")

    # conv3
    bw = 512
    _filter = (k_r * bw) // 256
    x = dual_path_block(x, _filter, _filter, bw, inc_sec[1], groups, 'down', b, name="conv3_1")
    for i in range(2, k_sec[1] + 1):
        x = dual_path_block(x, _filter, _filter, bw, inc_sec[1], groups, 'normal', b, name=f"conv3_{i}")

    # conv4
    bw = 1024
    _filter = (k_r * bw) // 256
    x = dual_path_block(x, _filter, _filter, bw, inc_sec[2], groups, 'down', b, name="conv4_1")
    for i in range(2, k_sec[1] + 1):
        x = dual_path_block(x, _filter, _filter, bw, inc_sec[2], groups, 'normal', b, name=f"conv4_{i}")

    # conv5
    bw = 2048
    _filter = (k_r * bw) // 256
    x = dual_path_block(x, _filter, _filter, bw, inc_sec[3], groups, 'down', b, name="conv5_1")
    for i in range(2, k_sec[1] + 1):
        x = dual_path_block(x, _filter, _filter, bw, inc_sec[3], groups, 'normal', b, name=f"conv5_{i}")

    x = layers.Concatenate(name="concat_last")([x[0], x[1]])
    x = layers.GlobalAvgPool2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)
    return keras.Model(inputs=_input, outputs=x, name=name)

