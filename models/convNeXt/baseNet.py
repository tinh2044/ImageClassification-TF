from numpy import random

import numpy as np
from keras.api import ops, initializers
from keras.src import layers
from keras.src.models import Sequential
import keras


class StochasticDepth(layers.Layer):

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (ops.shape(x)[0],) + (1,) * (len(ops.shape(x)) - 1)
            random_tensor = keep_prob + random.uniform(shape, 0, 1)
            random_tensor = ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class LayerScale(layers.Layer):

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, _):
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def ConvNeXtBlock(inputs, projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None):
    x = inputs

    x = layers.Conv2D(
        filters=projection_dim,
        kernel_size=7,
        padding="same",
        groups=projection_dim,
        name=f"{name}.dwconv",
    )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}.layernorm")(x)
    x = layers.Dense(4 * projection_dim, name=f"{name}.pwconv_1")(x)
    x = layers.Activation("gelu", name=f"{name}.gelu")(x)
    x = layers.Dense(projection_dim, name=f"{name}.pwconv_2")(x)

    if layer_scale_init_value is not None:
        x = LayerScale(
            layer_scale_init_value,
            projection_dim,
            name=f"{name}.layerScale",
        )(x)
    if drop_path_rate:
        x = StochasticDepth(
            drop_path_rate, name=f"{name}.stochasticDepth"
        )(x)
    else:
        x = layers.Activation("linear", name=f"{name}.identity")(x)

    return x


def ConvNeXt(
        num_cls,
        depths,
        projection_dims,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        name="convnext",
        input_shape=None,
):
    _input = layers.Input(shape=input_shape, name='_input')

    stem = Sequential(
        [
            layers.Conv2D(
                projection_dims[0],
                kernel_size=4,
                strides=4,
                name="stem_conv",
            ),
            layers.LayerNormalization(
                epsilon=1e-6, name="stem.layernorm"
            ),
        ],
        name="stem",
    )

    downsample_layers = [stem]

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-6,
                    name=f"{name}.down.layernorm" + str(i),
                ),
                layers.Conv2D(
                    projection_dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    name=f"{name}.down.conv" + str(i),
                ),
            ],
            name=f"{name}.down.block" + str(i),
        )
        downsample_layers.append(downsample_layer)

    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))
    ]

    cur = 0
    x = _input
    for i in range(len(projection_dims)):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtBlock(x,
                              projection_dim=projection_dims[i],
                              drop_path_rate=depth_drop_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value,
                              name=f"stage{i}.block{j}",
                              )
        cur += depths[i]

    x = layers.GlobalAveragePooling2D(name="head.gap")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="head.layernorm")(x)
    x = layers.Dense(num_cls, activation="softmax", name="head.output")(x)

    model = keras.Model(inputs=_input, outputs=x, name=name)

    return model


