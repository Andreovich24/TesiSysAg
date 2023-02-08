import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image


class RAFDB(data.Dataset):

    def __init__(self, split='train', dataset='rafdb', transform=None):
        self.transform = transform
        self.split = split
        self.dataset = dataset

        if self.dataset == "rafdb":
            if self.split == "train":
                self.data = pd.read_csv("../dataset/rafdb/train.csv")
                self.images = "../dataset/rafdb/train/"
            elif self.split == "test":
                self.data = pd.read_csv("../dataset/rafdb/test.csv")
                self.images = "../dataset/rafdb/test/"
        else:
            if self.split == "train":
                self.data = pd.read_csv("../dataset/rafdb_aligned/train.csv")
                self.images = "../dataset/rafdb_aligned/train/"
            elif self.split == "test":
                self.data = pd.read_csv("../dataset/rafdb_aligned/test.csv")
                self.images = "../dataset/rafdb_aligned/test/"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.loc[idx, "label"]
        label = int(label) - 1

        img_name = self.data.loc[idx, "image"]

        if self.dataset == "rafdb":
            img = Image.open(self.images + img_name)
        else:
            img = Image.open(self.images + img_name.replace(".jpg", "_aligned.jpg"))

        if self.transform is not None:
            img = self.transform(img)

        # return {'image': img, 'label': label}
        return {'image': img, 'label': label, 'name': img_name}


if __name__ == "__main__":
    split = "train"
    rafdb_train = RAFDB(split=split, dataset="rafdb")
    print("RAFDB {} set loaded".format(split))
    print("{} samples".format(len(rafdb_train)))

    for i in range(3):
        print(rafdb_train[i]["label"])
        rafdb_train[i]["image"].show()

