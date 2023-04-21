import torch.nn as nn
import torch.optim as optim
from common.config import settings
def optimizers(models):
    optimizer = optim.SGD(models.parameters(), lr=settings.LEARNING_LATE)
    return optimizer

def loss_fn():
    return nn.CrossEntropyLoss()