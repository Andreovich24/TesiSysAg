import numpy
import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import dlib


class LFW(data.Dataset):

    def __init__(self, split='train', masked=False, transform=None):
        self.transform = transform
        self.split = split
        self.masked = masked
        self.face_landmark = dlib.shape_predictor("../dataset/fer2013_mask/shape_predictor_68_face_landmarks.dat")
        # self.face_landmark = dlib.shape_predictor("../../fer2013_mask/shape_predictor_68_face_landmarks.dat")

        if masked is False:
            if self.split == "train":
                self.images_path = "../dataset/LFW/LFW-FER/train/"
                # self.images_path = "./train/"
                self.data = pd.read_csv("../dataset/LFW/LFW-FER/train.csv")
                # self.data = pd.read_csv("./train.csv")
            elif self.split == "val":
                self.images_path = "../dataset/LFW/LFW-FER/val/"
                # self.images_path = "./val/"
                self.data = pd.read_csv("../dataset/LFW/LFW-FER/val.csv")
                # self.data = pd.read_csv("./val.csv")
        else:
            if self.split == "train":
                self.images_path = "../dataset/LFW/M-LFW-FER/train/"
                # self.images_path = "./train/"
                self.data = pd.read_csv("../dataset/LFW/M-LFW-FER/train.csv")
                # self.data = pd.read_csv("./train.csv")
            elif self.split == "val":
                self.images_path = "../dataset/LFW/M-LFW-FER/val/"
                # self.images_path = "./val/"
                self.data = pd.read_csv("../dataset/LFW/M-LFW-FER/val.csv")
                # self.data = pd.read_csv("./val.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "image"]
        img = Image.open(self.images_path + img_name)
        img = numpy.asarray(img)

        label = self.data.loc[idx, "emotion"]
        label = int(label)

        x_face = int(self.data.loc[idx, "x"])
        y_face = int(self.data.loc[idx, "y"])
        w_face = int(self.data.loc[idx, "w"])
        h_face = int(self.data.loc[idx, "h"])
        img = img[y_face:y_face+h_face, x_face:x_face+w_face, :]

        if self.masked is True:
            # TODO: cambiare la dimensione del rectangle
            # rect = dlib.rectangle(0, 0, 250, 250)
            rect = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
            lds = self.face_landmark(img, rect)
            lds = self._shape_to_np(lds)
            x = max(lds[0][0], 0)
            # y = max(lds[19][1], lds[24][1], 0)
            y = 0
            w = lds[16][0] - x
            h = lds[30][1] - y
            img = img[y:y+h, x:x+w, :]

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
    split = "val"
    lfw_train = LFW(split=split, masked=True)
    print("LFW {} set loaded".format(split))
    print("{} samples".format(len(lfw_train)))

    for i in range(len(lfw_train)):
        elem = lfw_train[i]
        # print(elem["label"])
        #elem["image"].show()
        elem["image"].save("./images/" + elem["name"])