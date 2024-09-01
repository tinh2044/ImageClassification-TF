from models.resNet.baseNet import stack_residual_v1, res_net
import keras


def ResNet50(num_cls):

    def arch_builder(x):
        x = stack_residual_v1(x, 64, 3, 1, name="res_1")
        x = stack_residual_v1(x, 128, 4, 1, name="res_2")
        x = stack_residual_v1(x, 256, 6, 1, name="res_3")
        x = stack_residual_v1(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_cls,
        arch_builder,
        False,
        True,
        model_name="ResNet50", )


def ResNet101(num_cls):

    def arch_builder(x):
        x = stack_residual_v1(x, 64, 3, 1, name="res_1")
        x = stack_residual_v1(x, 128, 4, 1, name="res_2")
        x = stack_residual_v1(x, 256, 23, 1, name="res_3")
        x = stack_residual_v1(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_cls,
        arch_builder,
        False,
        True,
        model_name="ResNet101", )


def ResNet152(num_cls):
   

    def arch_builder(x):
        x = stack_residual_v1(x, 64, 3, 1, name="res_1")
        x = stack_residual_v1(x, 128, 8, 1, name="res_2")
        x = stack_residual_v1(x, 256, 36, 1, name="res_3")
        x = stack_residual_v1(x, 512, 3, 1, name="res_4")

        return x

    return res_net(
        num_cls,
        arch_builder,
        False,
        True,
        model_name="ResNet152", )


