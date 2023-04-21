from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from common.config import settings
class EfficientNetPretrain(nn.Module):
    """
        EfficientNet 사전학습 output_layer만 우리에게 맞게 수정
    """
    def __init__(self, version):
        super(EfficientNetPretrain, self).__init__()
        self.model = EfficientNet.from_pretrained(version,)
        self.outlayer = nn.Linear(1000, len(settings.TARGET))

    def forward(self, x):
        x = self.model(x)
        output = self.outlayer(x)
        return output