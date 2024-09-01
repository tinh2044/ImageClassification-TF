from models.denseNet.baseNet import dense_net


def DenseNet121(num_cls, grown_rate=32, reduction=0.5):
    return dense_net(
        num_cls,
        [6, 12, 24, 16],
        grown_rate,
        reduction,
        name="DenseNet121", )


def DenseNet169(num_cls, grown_rate=32, reduction=0.5):
    return dense_net(
        num_cls,
        [6, 12, 32, 32],
        grown_rate,
        reduction,
        name="DenseNet169", )


def DenseNet201(num_cls, grown_rate=32, reduction=0.5):
    return dense_net(
        num_cls,
        [6, 12, 48, 32],
        grown_rate,
        reduction,
        name="DenseNet201")


if __name__ == "__main__":
    model = DenseNet201(1000)
    print(model.summary())
