from math import ceil
import keras
from keras import layers


def ghost_module(x, _filter, ratio, conv_kernel, dw_kernel):
    conv_out_channel = ceil(_filter * 1.0 / ratio)

    conv = layers.Conv2D(int(conv_out_channel), conv_kernel, use_bias=False,
                         strides=(1, 1), padding='same', activation=None)(x)

    if ratio == 1:
        return conv

    dw = layers.DepthwiseConv2D(dw_kernel, 1, padding='same', use_bias=False,
                                depth_multiplier=ratio - 1, activation=None)(conv)

    slice_dw = layers.Lambda(lambda x: x[:, :, :, :int(_filter - conv_out_channel)])(dw)

    output = layers.Concatenate()([conv, slice_dw])

    return output


def se_module(inputs, filters, ratio):

    x = layers.GlobalAveragePooling2D()(inputs)

    x = layers.Reshape((1, 1, filters))(x)

    x = layers.Conv2D(filters // ratio, (1, 1), strides=(1, 1), padding='same',
                      use_bias=False, activation=None)(x)
    x = layers.Activation('relu')(x)

    excitation = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                               use_bias=False, activation=None)(x)
    excitation = layers.Activation('hard_sigmoid')(excitation)

    output = layers.Multiply()([inputs, excitation])

    return output


def gbneck(inputs, dw_kernel, strides, exp, out, ratio, use_se):

    x = layers.DepthwiseConv2D(dw_kernel, strides, padding='same', depth_multiplier=ratio - 1,
                               use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(out, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    y = ghost_module(inputs, exp, ratio, 1, 3)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    if strides > 1:
        y = layers.DepthwiseConv2D(dw_kernel, strides, padding='same', depth_multiplier=ratio - 1,
                                   use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)

    if use_se:
        y = se_module(y, exp, ratio)

    y = ghost_module(y, out, ratio, 1, 3)
    y = layers.BatchNormalization()(y)

    output = layers.Add()([x, y])

    return output


def GhostNet(num_cls, name="ghostNet", input_shape=(224, 224, 3)):

    _input = layers.Input(shape=input_shape, name="_input")

    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False)(_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    dw_kernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
    outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
    ratios = [2] * 16
    use_ses = [False, False, False, True, True, False, False, False, False, True, True, True, False, True, False, True]

    for i in range(16):
        x = gbneck(x, dw_kernels[i], strides[i], exps[i], outs[i], ratios[i], use_ses[i])

    x = layers.Conv2D(960, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 960))(x)

    x = layers.Conv2D(1280, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_cls, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.GlobalAvgPool2D(name="avg_pool")(x)
    x = layers.Softmax()(x)

    model = keras.Model(inputs=_input, outputs=x, name=name)

    return model

if __name__ == "__main__":
    model = GhostNet(1000)
    model.summary()