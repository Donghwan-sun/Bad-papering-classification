import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from common.config import settings

def optimizers(models):
    optimizer = optim.SGD(models.parameters(), lr=settings.LEARNING_LATE)
    return optimizer

def loss_fn():
    return nn.CrossEntropyLoss()

def save_models(best_model, epoch):
    folder = os.path.join(settings.MODEL_PATH, datetime.today().strftime(settings.DATE_FORMAT))

    if os.path.isdir(folder):
        pass
    else:
        os.makedirs(folder)
    torch.save(best_model, os.path.join(folder,
                                        f"{settings.PRETRAINED_VERSION}"
                                        f"-{epoch}"
                                        f"{settings.MODEL_EXTENSION}"))


folder = os.path.join(settings.MODEL_PATH, datetime.today().strftime(settings.DATE_FORMAT))
print(folder)