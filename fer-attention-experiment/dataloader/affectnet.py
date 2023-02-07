import torch.utils.data as data
import pandas as pd
from PIL import Image


class AffectNet(data.Dataset):

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split

        if self.split == "train":
            self.data = pd.read_csv("../dataset/affectnet/train.csv")
            self.images = "../dataset/affectnet/images/"
        elif self.split == "val":
            self.data = pd.read_csv("../dataset/affectnet/val.csv")
            self.images = "../dataset/affectnet/images/"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.loc[idx, "expression"]
        label = int(label)

        img_name = self.data.loc[idx, "subDirectory_filePath"]

        img = Image.open(self.images + img_name)

        face_x = self.data.loc[idx, "face_x"]
        face_y = self.data.loc[idx, "face_y"]
        face_width = self.data.loc[idx, "face_width"]
        face_height = self.data.loc[idx, "face_height"]

        img = img.crop((face_x, face_y, face_x+face_width, face_y+face_height))

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label}


if __name__ == "__main__":
    split = "train"
    affectnet_train = AffectNet(split=split)
    print("AffectNet {} set loaded".format(split))
    print("{} samples".format(len(affectnet_train)))

    for i in range(3):
        print(affectnet_train[i]["label"])
        affectnet_train[i]["image"].show()

