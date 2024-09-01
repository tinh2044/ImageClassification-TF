from models.nasNet.baseNet import nas_net


def NASNetMobile(num_cls):

    return nas_net(
        num_cls,
        name="NASNetMobile",
        penultimate_filters=1056,
        num_blocks=4,
        stem_block_filters=32,
        skip_reduction=False,
        filter_multiplier=2)


def NASNetLarge(num_cls):

    return nas_net(
        num_cls,
        name="NASNetLarge",
        penultimate_filters=4032,
        num_blocks=6,
        stem_block_filters=96,
        skip_reduction=True,
        filter_multiplier=2, )

if __name__ == "__main__":
    model = NASNetMobile(10)