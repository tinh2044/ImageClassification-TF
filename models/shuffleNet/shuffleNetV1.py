import keras
from keras import layers
import tensorflow as tf


def group_conv(x, in_filter, out_filter, kernel_size, strides, groups, name):
    if groups == 1:
        return layers.Conv2D(out_filter, kernel_size=kernel_size,
                             strides=strides, padding='same', name=f"{name}.g1_conv")(x)

    filter_per_group = in_filter // groups

    list_group = []
    b, h, w, c = x.shape
    for i in range(groups):
        group = layers.Lambda(lambda z: z[:, :, :, i * filter_per_group:(i + 1) * filter_per_group],
                              name=f"{name}.slice{i + 1}", output_shape=(h, w, filter_per_group))(x)
        list_group.append(
            layers.Conv2D(int(0.5 + out_filter / groups), kernel_size=kernel_size,
                          strides=strides, padding='same', name=f"{name}.g{i + 1}_conv")(group)

        )

    return layers.Concatenate(name=f"{name}.concat")(list_group)


def channels_shuffle(x, groups):
    b, h, w, c = x.shape.as_list()
    if c % groups != 0:
            raise ValueError(f"Number of channels {c} must be divisible by groups {groups}.")

    x = tf.reshape(x, (-1, h, w, groups, c//groups))
    x = tf.transpose(x, (0, 1, 2, 4, 3))
    x = tf.reshape(x, (-1, h, w, c))
    return x


def shuffle_unit(inputs, in_filter, out_filter, groups, strides, bottleneck_ratio, name):
    bottleneck_channels = int(bottleneck_ratio * out_filter)
    x = group_conv(inputs, in_filter, bottleneck_channels, 1, 1, groups, name + ".gconv1")
    x = layers.BatchNormalization(name=f"{name}.gconv1_bn")(x)
    x = layers.ReLU(name=f"{name}.conv1_relu")(x)

    x = layers.Lambda(lambda x: channels_shuffle(x, groups),
                      output_shape=x.shape[1:],
                      name=f"{name}.channels_shuffle")(x)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding="same", use_bias=False,
                               name=f"{name}.dwconv")(x)
    x = layers.BatchNormalization(name=f"{name}.dwconv.bn")(x)
    x = group_conv(x, bottleneck_channels,
                   out_filter=out_filter if strides == 1 else out_filter - in_filter,
                   groups=groups,
                   kernel_size=1,
                   strides=1,
                   name=f"{name}.gconv2")

    x = layers.BatchNormalization(name=f'{name}.gconv.bn')(x)
    if strides == 1:
        x = layers.Add(name=f'{name}.add')([x, inputs])
    else:
        avg = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f'{name}avg_pool')(inputs)
        x = layers.Concatenate(name=f'{name}.concat')([x, avg])
    return x


def shuffle_block(x, in_filter, out_filter, bottleneck_ratio, repeat, groups, name):
    x = shuffle_unit(x, in_filter=in_filter, out_filter=out_filter,
                     groups=groups, strides=2, bottleneck_ratio=bottleneck_ratio, name=f"{name}_1")

    for i in range(2, repeat + 1):
        x = shuffle_unit(x, in_filter=out_filter, out_filter=out_filter,
                         groups=groups, strides=1, bottleneck_ratio=bottleneck_ratio, name=f"{name}_{i}")

    return x


def cal_shuffle_map(groups, scale):
    out_filer = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

    exp = [1, 2, 4]

    return [int(out_filer[groups] * scale * i) for i in exp]


def shuffle_net_v1(num_cls, scale=1.0, groups=1, bottleneck_ratio=0.25):
    shuffle_map = cal_shuffle_map(groups, scale)
    shuffle_map.insert(0, 24)
    _input = layers.Input(shape=(224, 224, 3), name="input")
    x = layers.Conv2D(shuffle_map[0], kernel_size=3, strides=2, padding='same', name="stage1")(_input)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name="stage1.pool")(x)

    x = shuffle_block(x, in_filter=shuffle_map[0], out_filter=shuffle_map[1],
                      repeat=4, bottleneck_ratio=bottleneck_ratio, groups=groups, name="stage2")

    x = shuffle_block(x, in_filter=shuffle_map[1], out_filter=shuffle_map[2],
                      repeat=8, bottleneck_ratio=bottleneck_ratio, groups=groups, name="stage3")

    x = shuffle_block(x, in_filter=shuffle_map[2], out_filter=shuffle_map[3],
                      repeat=4, bottleneck_ratio=bottleneck_ratio, groups=groups, name="stage4")

    x = layers.GlobalAvgPool2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)
    return keras.Model(inputs=_input, outputs=x, name=f"ShuffleNetV1_{scale}_{groups}")


def ShuffleNetV1_0_5x(num_cls):
    return shuffle_net_v1(num_cls, 0.5, 3)


def ShuffleNetV1_1_0x(num_cls):
    return shuffle_net_v1(num_cls, 1.5, 3)


def ShuffleNetV1_1_5x(num_cls):
    return shuffle_net_v1(num_cls, 1.5, 3)


def ShuffleNetV1_2_0x(num_cls):
    return shuffle_net_v1(num_cls, 2.0, 3)
