import keras

from models.convMixer import ConvMixer
from models.convNeXt import ConvNeXtSmall, ConvNeXtTiny, ConvNeXtLarge, ConvNeXtXLarge, ConvNeXtBase
from models.denseNet import DenseNet121, DenseNet169, DenseNet201
from models.DPN import DPN92, DPN98, DPN131
from models.efficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
from models.efficientNet import EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from models.inceptionNet import InceptionNetV1, InceptionNetV3
from models.inceptionResNet import InceptionResNetV1, InceptionResNetV2
from models.mobileNet import MobileNetV1, MobileNetV3Large, MobileNetV3Small, MobileNetV2
from models.ghostNet import GhostNet
from models.nasNet import NASNetLarge, NASNetMobile
from models.resNet import ResNet50, ResNet101, ResNet152, ResNetV2_50, ResNetV2_101, ResNetV2_152
from models.shuffleNet import ShuffleNetV1_0_5x, ShuffleNetV1_1_0x, ShuffleNetV1_1_5x, ShuffleNetV1_2_0x
from models.shuffleNet import ShuffleNetV2_0_5x, ShuffleNetV2_1_0x, ShuffleNetV2_1_5x, ShuffleNetV2_2_0x
from models.vgg import VGG16, VGG19
from models.xception import Xception

MODEL_NAMES = [
    "ConvMixer",
    "ConvNeXtSmall", "ConvNeXtTiny", "ConvNeXtLarge", "ConvNeXtXLarge", "ConvNeXtBase",
    "DenseNet121", "DenseNet169", "DenseNet201",
    "DPN92", "DPN98", "DPN131",
    "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
    "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7",
    "InceptionNetV1", "InceptionNetV3",
    "InceptionResNetV1", "InceptionResNetV2",
    "MobileNetV1", "MobileNetV3Large", "MobileNetV3Small", "MobileNetV2",
    "GhostNet", "NASNetMobile", "NASNetLarge",
    "ResNet50", "ResNet101", "ResNet152", "ResNetV2_50", "ResNetV2_101", "ResNetV2_152",
    "ShuffleNetV1_0_5x", "ShuffleNetV1_1_0x", "ShuffleNetV1_1_5x", "ShuffleNetV1_2_0x",
    "ShuffleNetV2_0_5x", "ShuffleNetV2_1_0x", "ShuffleNetV2_1_5x", "ShuffleNetV2_2_0x",
    "VGG16", "VGG19",
    "Xception"
]


def get_model(name, num_cls) -> keras.Model:
    if name not in MODEL_NAMES:
        raise ValueError(f"{name} is not in {MODEL_NAMES}")
    return globals()[name](num_cls)
