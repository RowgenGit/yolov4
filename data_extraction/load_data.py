import pandas as pd
import numpy as np
import cv2


def load_csv_file(path, classes):
    images_and_boxes = pd.read_csv(path)
    #images_and_boxes.replace({"label": classes}, inplace=True)
    images_and_boxes['bboxes'] = images_and_boxes.drop("image", axis=1).values.tolist()
    images_and_boxes.drop(['xmin', 'ymin', 'xmax', 'ymax', 'label'], axis=1, inplace=True)
    images_and_boxes = images_and_boxes.groupby("image").agg(lambda x: list(x)).reset_index()
    return images_and_boxes


def get_sample(data_path, df, index):
    image_path = data_path + df.iloc[index].loc["image"]
    image = cv2.imread(image_path)
    boxes = np.array(df.iloc[index].loc["bboxes"])
    return image, boxes


def read_classes(path):
    names = {}
    with open(path, 'r') as data:
        for ID, name in enumerate(data):
            name = name.strip('\n')
            names[name] = ID
    return names
