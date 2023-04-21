from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from common.config import settings
from data.preprocess import transform, split_dataset

class CustomDataset(Dataset):
    """
        도배하자 분류 커스텀 데이터셋
    """
    def __init__(self, path, transform=None):
        self.dataset = glob(path + "\\*\\*")
        self.transform = transform
    def __getitem__(self, index):
        """
        :param index (int): 꺼내고자하는 인덱스의 값
        :return:
                image (PIL Image or torch):이미지 배열 값
                target (string): 라벨 값
        """
        item = self.dataset[index]
        image = Image.open(item).convert("RGB")
        target = item.split("\\")[-2]
        target = settings.TARGET.index(target)

        if self.transform:
            image = self.transform(image)

        result = {"image": image, "target": torch.tensor(target, dtype=torch.float16)}
        return result
    def __len__(self):
        return len(self.dataset)

class DatasetVisualization:
    def __init__(self, dataset):
        """
        sample:
            cd = CustomDataset(settings.TRAIN_DATA_PATH)
            image_viewer = DatasetVisualization(cd)
            image_viewer.visual(122)
        :param dataset: torch의 dataset 객체
        """
        self.dataset = dataset
    def visual(self, index):
        """
            이미지 데이터셋 시각화 메소드
        :param index (int): 보고싶은 이미지 index 값
        :return:
        """
        image = self.dataset[index]["image"]
        image = np.asarray(image)
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.show()

def dataloader_init():
    dataloader_dict = {}
    train_dataset = CustomDataset(settings.TRAIN_DATA_PATH, transform("train"))
    dataset_dict = split_dataset(len(train_dataset),
                                 settings.TRAIN_RATIO,
                                 settings.VALIDATION_RATIO,
                                 settings.TEST_RATIO)
    test_dataset = None

    if "test" in list(dataset_dict.keys()):
        train_dataset, validation_dataset, test_dataset = random_split(train_dataset,
                                                                       [dataset_dict["train"],
                                                                        dataset_dict["validation"],
                                                                        dataset_dict["test"]]
                                                                       )
    else:
        train_dataset, validation_dataset = random_split(train_dataset,
                                                         [dataset_dict["train"],
                                                          dataset_dict["validation"]])

    dataloader_dict["train"] = DataLoader(train_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=settings.SHUFFLE)

    dataloader_dict["validation"] = DataLoader(validation_dataset,
                                               batch_size=settings.BATCH_SIZE,
                                               shuffle=settings.SHUFFLE)
    if test_dataset:
        dataloader_dict["test"] = DataLoader(test_dataset,
                                             batch_size=settings.BATCH_SIZE,
                                             shuffle=settings.SHUFFLE)

    return dataloader_dict



