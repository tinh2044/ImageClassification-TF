from keras import layers
import keras
from models.resNet.baseNet import res_net, stack_residual_v2


def ResNetV2_50(num_cls):

    def arch_builder(x):
        x = stack_residual_v2(x, 64, 3, 1, name="res_1")
        x = stack_residual_v2(x, 128, 4, 1, name="res_2")
        x = stack_residual_v2(x, 256, 6, 1, name="res_3")
        x = stack_residual_v2(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_cls,
        arch_builder,
        False,
        True,
        model_name="ResNetV2_50", )


def ResNetV2_101(num_cls):

    def arch_builder(x):
        x = stack_residual_v2(x, 64, 3, 1, name="res_1")
        x = stack_residual_v2(x, 128, 4, 1, name="res_2")
        x = stack_residual_v2(x, 256, 23, 1, name="res_3")
        x = stack_residual_v2(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_cls,
        arch_builder,
        False,
        True,
        model_name="ResNetV2_101", )


def ResNetV2_152(num_cls):

    def arch_builder(x):
        x = stack_residual_v2(x, 64, 3, 1, name="res_1")
        x = stack_residual_v2(x, 128, 8, 1, name="res_2")
        x = stack_residual_v2(x, 256, 36, 1, name="res_3")
        x = stack_residual_v2(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_cls,
        arch_builder,
        False,
        True,
        model_name="ResNetV2_152", )
