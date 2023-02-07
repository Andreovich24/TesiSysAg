import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image


class FER2013(data.Dataset):

    def __init__(self, split='train', dataset='fer2013', transform=None, as_rgb=False):
        self.transform = transform
        self.split = split
        self.as_rgb = as_rgb
        self.dataset = dataset

        if self.dataset == "fer2013clean":
            if self.split == "train":
                self.data = pd.read_csv("../dataset/fer2013_clean/train.csv")
            elif self.split == "val":
                self.data = pd.read_csv("../dataset/fer2013_clean/val.csv")
            elif self.split == "test":
                self.data = pd.read_csv("../dataset/fer2013_clean/test.csv")
        else:
            if self.split == "train":
                self.data = pd.read_csv("../dataset/fer2013/train.csv")
            elif self.split == "val":
                self.data = pd.read_csv("../dataset/fer2013/val.csv")
            elif self.split == "test":
                self.data = pd.read_csv("../dataset/fer2013/test.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.loc[idx, "pixels"]
        img = img.split(" ")
        img = np.array(img, 'int')
        img = img.reshape(48, 48)

        label = self.data.loc[idx, "emotion"]
        label = int(label)

        if self.as_rgb is True:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label, 'name': str(idx)+".jpg"}


if __name__ == "__main__":
    split = "train"
    fer_train = FER2013(split=split, as_rgb=True)
    print("FER2013 {} set loaded".format(split))
    print("{} samples".format(len(fer_train)))

    for i in range(3):
        print(fer_train[i]["label"])
        fer_train[i]["image"].show()

