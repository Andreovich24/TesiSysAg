import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image

# Labels:
# 0: "Angry"
# 1: "Disgust"
# 2: "Fear"
# 3: "Happy"
# 4: "Sad"
# 5: "Surprise"
# 6: "Neutral"

class RAFDBFER2013(data.Dataset):

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split

        if self.split == "train":
            self.data_rafdb = pd.read_csv("../dataset/rafdb_aligned/train.csv")
            self.images_rafdb = "../dataset/rafdb_aligned/train/"
            # self.data_rafdb["dataset"] = ["rafdb"] * self.data_rafdb.shape[0]
            self.data_fer2013 = pd.read_csv("../dataset/fer2013/train.csv")
            # self.data_fer2013["dataset"] = ["fer2013"] * self.data_fer2013.shape[0]
            rafdb_idx = self.data_rafdb.index.tolist()
            fer2013_idx = self.data_fer2013.index.tolist()
            self.data = pd.DataFrame({
                "ref_id": rafdb_idx + fer2013_idx,
                "dataset": (["rafdb"] * self.data_rafdb.shape[0]) + (["fer2013"] * self.data_fer2013.shape[0])
            })
        elif self.split == "val":
            self.data_rafdb = pd.read_csv("../dataset/rafdb_aligned/test.csv")
            self.images_rafdb = "../dataset/rafdb_aligned/test/"
            self.data_fer2013 = pd.read_csv("../dataset/fer2013/val.csv")
            rafdb_idx = self.data_rafdb.index.tolist()
            fer2013_idx = self.data_fer2013.index.tolist()
            self.data = pd.DataFrame({
                "ref_id": rafdb_idx + fer2013_idx,
                "dataset": (["rafdb"] * self.data_rafdb.shape[0]) + (["fer2013"] * self.data_fer2013.shape[0])
            })
        elif self.split == "valrafdb":
            self.images_rafdb = "../dataset/rafdb_aligned/test/"
            self.data_rafdb = pd.read_csv("../dataset/rafdb_aligned/test.csv")
            rafdb_idx = self.data_rafdb.index.tolist()
            self.data = pd.DataFrame({
                "ref_id": rafdb_idx,
                "dataset": ["rafdb"] * self.data_rafdb.shape[0]
            })
        elif self.split == "valfer2013":
            self.data_fer2013 = pd.read_csv("../dataset/fer2013/val.csv")
            fer2013_idx = self.data_fer2013.index.tolist()
            self.data = pd.DataFrame({
                "ref_id": fer2013_idx,
                "dataset": (["fer2013"] * self.data_fer2013.shape[0])
            })
        elif self.split == "test":
            self.data_fer2013 = pd.read_csv("../dataset/fer2013/test.csv")
            fer2013_idx = self.data_fer2013.index.tolist()
            self.data = pd.DataFrame({
                "ref_id": fer2013_idx,
                "dataset": (["fer2013"] * self.data_fer2013.shape[0])
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ref_id = self.data.loc[idx, "ref_id"]
        dataset = self.data.loc[idx, "dataset"]

        if dataset == "rafdb":
            label = self.data_rafdb.loc[ref_id, "label"]
            label = int(label) - 1
            label = self.convert_label(label)

            img_name = self.data_rafdb.loc[ref_id, "image"]

            img = Image.open(self.images_rafdb + img_name.replace(".jpg", "_aligned.jpg"))

            if self.transform is not None:
                img = self.transform(img)

            # return {'image': img, 'label': label}
            return {'image': img, 'label': label, 'name': img_name}

        else:
            img = self.data_fer2013.loc[ref_id, "pixels"]
            img = img.split(" ")
            img = np.array(img, 'int')
            img = img.reshape(48, 48)

            label = self.data_fer2013.loc[ref_id, "emotion"]
            label = int(label)

            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis = 2)

            img = Image.fromarray(img.astype(np.uint8))

            if self.transform is not None:
                img = self.transform(img)

            return {'image': img, 'label': label, 'name': str(ref_id) + ".jpg"}

    # convert lables from RAF-DB to FER 2013
    def convert_label(self, label):
        label_mapping = {0: 5,
                         1: 2,
                         2: 1,
                         3: 3,
                         4: 4,
                         5: 0,
                         6: 6}
        return label_mapping[label]


if __name__ == "__main__":
    split = "train"
    rafdbfer2013_train = RAFDBFER2013(split=split)
    print("RAFDB+FER2013 {} set loaded".format(split))
    print("{} samples".format(len(rafdbfer2013_train)))

    for i in range(len(rafdbfer2013_train)):
        print(rafdbfer2013_train[i]["label"])
    #     rafdb_train[i]["image"].show()

