from train.trainer import Trainner
from data.loader import dataloader_init
from common.config import settings
from common.utils import device
from models.utils import optimizers, loss_fn
from models import models

if __name__ == "__main__":
    dataloader_dict = dataloader_init()
    model = models.EfficientNetPretrain(settings.PRETRAINED_VERSION)

    trainner = Trainner(train_dataloader=dataloader_dict["train"],
                        validation_loader=dataloader_dict["validation"],
                        device=device(),
                        optimizer=optimizers(model),
                        model=model,
                        loss_fn=loss_fn())
    trainner.fit()