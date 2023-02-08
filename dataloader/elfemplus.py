import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image


class ElfemPlus(data.Dataset):

    def __init__(self, split='train', transform=None, as_rgb=False):
        self.transform = transform
        self.split = split
        self.as_rgb = as_rgb

        if self.split == "train":
            self.data = pd.read_csv("../dataset/elfem-plus/train.csv")
            self.images = "../dataset/elfem-plus/train/"
        elif self.split == "val":
            self.data = pd.read_csv("../dataset/elfem-plus/val.csv")
            self.images = "../dataset/elfem-plus/val/"
        elif self.split == "test":
            self.data = pd.read_csv("../dataset/elfem-plus/test.csv")
            self.images = "../dataset/elfem-plus/test/"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.loc[idx, "label"]
        label = int(label)

        img_name = self.data.loc[idx, "image"]

        img = Image.open(self.images + img_name)
        if self.as_rgb is True:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label}


if __name__ == "__main__":
    split = "train"
    elfemplus_train = ElfemPlus(split=split, as_rgb = True)
    print("ElfemPlus {} set loaded".format(split))
    print("{} samples".format(len(elfemplus_train)))

    for i in range(3):
        print(elfemplus_train[i]["label"])
        elfemplus_train[i]["image"].show()

