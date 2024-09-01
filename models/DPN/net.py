from models.DPN.baseNet import dpn


def DPN92(num_cls):
    return dpn(
        num_cls,
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), name="DPN92")


def DPN98(num_cls):
    return dpn(
        num_cls,
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128), name="DPN98")


def DPN131(num_cls):
    return dpn(
        num_cls,
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128), name="DPN131")


if __name__ == "__main__":
    net = DPN92(10)
    print(net)
