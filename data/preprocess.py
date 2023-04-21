import torch
from torchvision import transforms
from common.config import settings

def transform(type=None):
    if type is None:
        trans = None
    if type == "train":
        trans = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor()
        ])

    elif type == "validation":
        trans = transforms.Compose([
            transforms.Resize(settings.IMAGE_SIZE),
            transforms.ToTensor()
        ])
    return trans

def split_dataset(dataset_size, train_ratio, validation_ratio, test_ratio=None):
    dataset_dict = {}
    train_size = int(dataset_size * train_ratio)
    dataset_dict["train"] = train_size

    if train_ratio + validation_ratio != 1.0 and test_ratio:
        validation_size = int(dataset_size * validation_ratio)
        test_size = dataset_size - train_size - validation_size
        dataset_dict["test"] = test_size
    else:
        validation_size = dataset_size - train_size

    dataset_dict["validation"] = validation_size

    return dataset_dict