from glob import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from common.config import settings

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

        if self.transform:
            image = self.transform(image)
        result = {"image": image, "target": target}
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
        img = np.array(image)
        plt.imshow(img)
        plt.show()
"""

"""