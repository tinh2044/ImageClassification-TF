from models.convNeXt.baseNet import ConvNeXt


def ConvNeXtTiny(num_cls, name="ConvNeXtTiny"):
    return ConvNeXt(
        num_cls=num_cls,
        depths=[3, 3, 9, 3],
        projection_dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        name=name,
        input_shape=(224, 224, 3))


def ConvNeXtSmall(num_cls, name="ConvNeXtSmall"):
    return ConvNeXt(
        num_cls=num_cls,
        depths=[3, 3, 27, 3],
        projection_dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        name=name,
        input_shape=(224, 224, 3)
    )


def ConvNeXtBase(num_cls, name="ConvNeXtBase"):
    return ConvNeXt(
        num_cls=num_cls,
        depths=[3, 3, 27, 3],
        projection_dims=[128, 256, 512, 1024],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        name=name,
        input_shape=(224, 224, 3)
    )


def ConvNeXtLarge(num_cls, name="ConvNeXtLarge"):
    return ConvNeXt(
        num_cls=num_cls,
        depths=[3, 3, 27, 3],
        projection_dims=[192, 384, 768, 1536],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        name=name,
        input_shape=(224, 224, 3)
    )


def ConvNeXtXLarge(num_cls, name="ConvNeXtXLarge"):
    return ConvNeXt(
        num_cls=num_cls,
        depths=[3, 3, 27, 3],
        projection_dims=[256, 512, 1024, 2048],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        name=name,
        input_shape=(224, 224, 3)
    )


if __name__ == "__main__":
    model = ConvNeXtTiny(1000)
    print(model.summary())
