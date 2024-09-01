import keras
from keras import layers
import tensorflow as tf


def channels_shuffle(x, groups):
    b, h, w, c = x.shape.as_list()
    if c % groups != 0:
            raise ValueError(f"Number of channels {c} must be divisible by groups {groups}.")

    x = tf.reshape(x, (-1, h, w, groups, c//groups))
    x = tf.transpose(x, (0, 1, 2, 4, 3))
    x = tf.reshape(x, (-1, h, w, c))
    return x



def channel_split(x):
    c = x.shape[-1]
    branch = x[:, :, :, :c // 2]
    x = x[:, :, :, c // 2:]
    return branch, x


def shuffle_block_s1(x, _filter, groups, name):
    branch, x = channel_split(x)

    x = layers.Conv2D(_filter // 2, kernel_size=1, strides=1, padding='same', name=f"{name}.1x1conv_1")(x)
    x = layers.BatchNormalization(name=f"{name}.1x1bn_1")(x)
    x = layers.ReLU(name=f"{name}.1x1relu_1")(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', name=f"{name}.dwconv")(x)
    x = layers.BatchNormalization(name=f"{name}.bn")(x)

    x = layers.Conv2D(_filter // 2, kernel_size=1, strides=1, padding='same', name=f"{name}.1x1conv_1")(x)
    x = layers.BatchNormalization(name=f"{name}.1x1bn_2")(x)
    x = layers.ReLU(name=f"{name}.1x1relu_2")(x)

    x = layers.Concatenate(name=f"{name}.concat")([branch, x])
    x = layers.Lambda(lambda y: channels_shuffle(y, groups), name=f"{name}.shuffle", output_shape=x.shape[1:])(x)
    # x = channels_shuffle(x, groups)
    return x


def shuffle_block_s2(x, _filter, groups, name):
    branch_1 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name=f"{name}.b1_dwconv")(x)
    branch_1 = layers.BatchNormalization(name=f"{name}.b1_bn")(branch_1)
    branch_1 = layers.Conv2D(_filter // 2, kernel_size=1, name=f"{name}.b1_1x1_1")(branch_1)
    branch_1 = layers.BatchNormalization(name=f"{name}.b1_1x1bn_1")(branch_1)
    branch_1 = layers.ReLU(name=f"{name}.b1_1x1relu_1")(branch_1)

    branch_1 = layers.Conv2D(_filter // 2, kernel_size=1, strides=1, padding='same', name=f"{name}.b1_1x1_2")(branch_1)
    branch_1 = layers.BatchNormalization(name=f"{name}.b1_1x1bn_2")(branch_1)
    branch_1 = layers.ReLU(name=f"{name}.b1_1x1relu_2")(branch_1)

    branch_2 = layers.Conv2D(_filter // 2, kernel_size=1, strides=1, padding='same', name=f"{name}.b2_1x1conv_1")(x)
    branch_2 = layers.BatchNormalization(name=f"{name}.1x1bn_1")(branch_2)
    branch_2 = layers.ReLU(name=f"{name}.1x1relu_1")(branch_2)

    branch_2 = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name=f"{name}.dwconv")(branch_2)
    branch_2 = layers.BatchNormalization(name=f"{name}.bn")(branch_2)

    branch_2 = layers.Conv2D(_filter // 2, kernel_size=3, strides=1, padding='same', name=f"{name}.b2_1x1conv_2")(
        branch_2)
    branch_2 = layers.BatchNormalization(name=f"{name}.1x1bn_2")(branch_2)
    branch_2 = layers.ReLU(name=f"{name}.1x1relu_2")(branch_2)

    x = layers.Concatenate(name=f"{name}.concat")([branch_1, branch_2])
    x = layers.Lambda(lambda y: channels_shuffle(y, groups), name=f"{name}.shuffle", output_shape=x.shape[1:])(x)

    return x


def shuffle_block(x, _filter, repeat, groups, name):
    x = shuffle_block_s2(x, _filter, groups, name=f"{name}.1_s2")

    for i in range(2, repeat + 1):
        x = shuffle_block_s2(x, _filter, groups, name=f"{name}.{i}_s2")

    return x


def shuffle_net_v2(num_cls, shuffle_map, name):
    _input = layers.Input(shape=(224, 224, 3), name="input")
    x = layers.Conv2D(24, kernel_size=3, strides=2, padding='same', name="stage1")(_input)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name="stage1.pool")(x)

    x = shuffle_block(x, _filter=shuffle_map[0],
                      repeat=4, groups=2, name="stage2")

    x = shuffle_block(x, _filter=shuffle_map[1],
                      repeat=4, groups=2, name="stage3")
    x = shuffle_block(x, _filter=shuffle_map[2],
                      repeat=4, groups=2, name="stage4")

    x = layers.Conv2D(shuffle_map[-1], 1, 1, padding='same', name="stage5")(x)
    x = layers.GlobalAvgPool2D(name="avg_pool")(x)
    x = layers.Dense(num_cls, activation="softmax", name="output")(x)
    return keras.Model(inputs=_input, outputs=x, name=name)


def ShuffleNetV2_0_5x(num_cls):
    return shuffle_net_v2(num_cls, [24, 48, 96, 192, 1024], "ShuffleNetV2_0_5x")


def ShuffleNetV2_1_0x(num_cls):
    return shuffle_net_v2(num_cls, [116, 232, 464, 1024], "ShuffleNetV2_1_0x")


def ShuffleNetV2_1_5x(num_cls):
    return shuffle_net_v2(num_cls, [176, 352, 704, 1024], "ShuffleNetV2_1_5x")


def ShuffleNetV2_2_0x(num_cls):
    return shuffle_net_v2(num_cls, [244, 488, 976, 2048], "ShuffleNetV2_2_0x")


if __name__ == "__main__":
    model = ShuffleNetV2_2_0x(1000)
    print(model.summary())
