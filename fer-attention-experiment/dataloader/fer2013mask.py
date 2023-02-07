import numpy
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import dlib


class FER2013Mask(data.Dataset):

    def __init__(self, split='train', masked=False, transform=None, as_rgb=False):
        self.transform = transform
        self.split = split
        self.as_rgb = as_rgb
        self.masked = masked
        self.face_landmark = dlib.shape_predictor("../dataset/fer2013_mask/shape_predictor_68_face_landmarks.dat")

        if self.split == "train":
            self.images_path = "../dataset/fer2013_mask/train/"
            self.data = pd.read_csv("../dataset/fer2013_mask/train.csv")
        elif self.split == "val":
            self.images_path = "../dataset/fer2013_mask/val/"
            self.data = pd.read_csv("../dataset/fer2013_mask/val.csv")
        elif self.split == "test":
            self.images_path = "../dataset/fer2013_mask/test/"
            self.data = pd.read_csv("../dataset/fer2013_mask/test.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "image"]
        img = Image.open(self.images_path + img_name)
        img = numpy.asarray(img)

        label = self.data.loc[idx, "emotions"]
        label = int(label)

        if self.masked is True:
            rect = dlib.rectangle(0, 0, 48, 48)
            lds = self.face_landmark(img, rect)
            lds = self._shape_to_np(lds)
            x = max(lds[0][0], 0)
            # y = max(lds[19][1], lds[24][1], 0)
            y = 0
            w = lds[16][0] - x
            h = lds[30][1] - y
            img = img[y:y+h, x:x+w]

        if self.as_rgb is True:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label, 'name': img_name}

    def _shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords


if __name__ == "__main__":
    split = "train"
    fer_train = FER2013Mask(split=split, as_rgb=True, masked=True)
    print("FER2013 {} set loaded".format(split))
    print("{} samples".format(len(fer_train)))

    for i in range(3):
        elem = fer_train[i]
        print("Emotion", elem["label"])
        elem["image"].show()
