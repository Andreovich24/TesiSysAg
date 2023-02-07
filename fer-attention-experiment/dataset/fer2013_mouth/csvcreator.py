import pandas as pd
import numpy as np
from PIL import Image

train_csv = "./train.csv"
val_csv = "./val.csv"
test_csv = "./test.csv"

fer2013_train_csv = "../fer2013/train.csv"
fer2013_val_csv = "../fer2013/val.csv"
fer2013_test_csv = "../fer2013/test.csv"


def get_class(data, fer2013data):
    data_csv = pd.read_csv(data)
    fer2013data_csv = pd.read_csv(fer2013data)
    emotions = []
    images = []

    for i, row in data_csv.iterrows():
        row_number = int(row["image"].split(".")[0])
        images.append(row["image"])
        emotions.append(fer2013data_csv.loc[row_number, "emotion"])
    data_dict = {
        "image": images,
        "emotions": emotions
    }
    df = pd.DataFrame(data_dict)
    df.to_csv(data, index=False)


get_class(train_csv, fer2013_train_csv)
get_class(val_csv, fer2013_val_csv)
get_class(test_csv, fer2013_test_csv)
