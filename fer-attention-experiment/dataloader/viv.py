import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image


class VIV(data.Dataset):

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split

        if self.split == "train":
            self.data = pd.read_csv("../dataset/viv/train.csv")
            # self.images = "../dataset/viv/TRAIN/"
            self.images = "../dataset/viv"
        elif self.split == "val":
            self.data = pd.read_csv("../dataset/viv/val.csv")
            # self.images = "../dataset/viv/VALIDATION/"
            self.images = "../dataset/viv"
        elif self.split == "test":
            self.data = pd.read_csv("../dataset/viv/test.csv")
            # self.images = "../dataset/viv/TEST/"
            self.images = "../dataset/viv"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.loc[idx, "label"]
        label = int(label)

        img_name = self.data.loc[idx, "image"]

        # img = Image.open(self.images + img_name)
        img = Image.open(self.images + img_name[1:])

        if self.transform is not None:
            img = self.transform(img)

        # return {'image': img, 'label': label}
        return {'image': img, 'label': label, 'name': img_name}


if __name__ == "__main__":
    split = "train"
    viv_train = VIV(split=split)
    print("VIV {} set loaded".format(split))
    print("{} samples".format(len(viv_train)))

    for i in range(3):
        print(viv_train[i]["label"])
        viv_train[i]["image"].show()

