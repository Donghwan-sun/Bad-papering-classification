from efficientnet_pytorch import EfficientNet

def EfficientNet_pretrain(version):
    return EfficientNet.from_pretrained(version)