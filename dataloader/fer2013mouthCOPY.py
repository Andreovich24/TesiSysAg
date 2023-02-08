import numpy
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import dlib


class FER2013Mouth(data.Dataset):

    def __init__(self, split='train', cropped_mouth=False, transform=None, as_rgb=False):
        self.transform = transform
        self.split = split
        self.as_rgb = as_rgb
        self.cropped_mouth = cropped_mouth
        self.face_landmark = dlib.shape_predictor("../dataset/fer2013_mouth/shape_predictor_68_face_landmarks.dat")

        if self.split == "train":
            self.images_path = "../dataset/fer2013_mouth/train/"
            self.data = pd.read_csv("../dataset/fer2013_mouth/train.csv")
        elif self.split == "val":
            self.images_path = "../dataset/fer2013_mouth/val/"
            self.data = pd.read_csv("../dataset/fer2013_mouth/val.csv")
        elif self.split == "test":
            self.images_path = "../dataset/fer2013_mouth/test/"
            self.data = pd.read_csv("../dataset/fer2013_mouth/test.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "image"]
        img = Image.open(self.images_path + img_name)
        img = numpy.asarray(img)

        label = self.data.loc[idx, "emotions"]
        label = int(label)

        if self.cropped_mouth is True:
            rect = dlib.rectangle(0, 0, 48, 48)
            lds = self.face_landmark(img, rect)
            lds = self._shape_to_np(lds)

            x = max(lds[2][0], 0)
            y = min(lds[2][1], lds[14][1])
            # y = round((lds[2][1]+lds[14][1])/2)  il punto medio tra il punto 2 e 14 per colonne

            w = lds[14][0] - x
            h = max(lds[6][1], lds[10][1]) - y

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
    fer_train = FER2013Mouth(split=split, as_rgb=True, cropped_mouth=True)
    print("FER2013 {} set loaded".format(split))
    print("{} samples".format(len(fer_train)))

    for i in range(3):
        elem = fer_train[i]
        print("Emotion", elem["label"])
        elem["image"].show()
        f = open('../result/Prova.txt', 'a')
        f.write(str(elem["label"])+'\n')
        f.close()
