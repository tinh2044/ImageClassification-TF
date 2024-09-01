from models.efficientNet.baseNet import EfficientNet


def EfficientNetB0(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(224, 224, 3),
        width_coefficient=1.0,
        depth_coefficient=1.0,
        drop_connect_rate=0.2,
        model_name="EfficientNetB0")


def EfficientNetB1(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(240, 240, 3),
        width_coefficient=1.0,
        depth_coefficient=1.1,
        drop_connect_rate=0.2,
        model_name="EfficientNetB1")


def EfficientNetB2(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(260, 260, 3),
        width_coefficient=1.1,
        depth_coefficient=1.2,
        drop_connect_rate=0.3,
        model_name="EfficientNetB2")


def EfficientNetB3(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(380, 380, 3),
        width_coefficient=1.2,
        depth_coefficient=1.4,
        drop_connect_rate=0.3,
        model_name="EfficientNetB3")


def EfficientNetB4(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(300, 300, 3),
        width_coefficient=1.4,
        depth_coefficient=1.8,
        drop_connect_rate=0.4,
        model_name="EfficientNetB4")


def EfficientNetB5(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(300, 300, 3),
        width_coefficient=1.6,
        depth_coefficient=2.2,
        drop_connect_rate=0.4,
        model_name="EfficientNetB5")


def EfficientNetB6(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(528, 528, 3),
        width_coefficient=1.8,
        depth_coefficient=2.6,
        drop_connect_rate=0.5,
        model_name="EfficientNetB6")


def EfficientNetB7(num_cls):
    return EfficientNet(
        num_cls=num_cls,
        input_shape=(600, 600, 3),
        width_coefficient=2.0,
        depth_coefficient=3.1,
        drop_connect_rate=0.5,
        model_name="EfficientNetB7")
