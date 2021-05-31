import os
import pandas as pd
from shutil import copyfile
from test import zip_dir, read_classes

img_path = "data/"
label_path = "output/"

classes = read_classes("classes.txt")

dest_folder = "training_data"
dest_path = dest_folder + "/"


if __name__ == "__main__":
    annotations = pd.DataFrame()
    image_list = []
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    label_list = []

    for sample in os.listdir(label_path):
        labels_dir = label_path + sample + "/"
        labels_file_list = os.listdir(labels_dir)
        img_dir = img_path + sample + "/"
        img_file_list = os.listdir(img_dir)

        for file in labels_file_list:
            if file[-4:] != ".txt":
                continue
            complete_file_name = labels_dir + file
            f = open(complete_file_name, "r")
            lines = f.readlines()
            for line in lines:
                data = line.split(" ")
                if len(data) < 2:
                    continue
                x1 = float(data[2])
                y1 = float(data[3])
                w = float(data[4])
                h = float(data[5])
                pt1 = (x1, y1)
                pt2 = (x1 + w, y1 + h)
                print("coordinates calculated")

                img_name = file[:-4] + ".jpg"
                complete_img_name = img_dir + img_name
                if img_name not in img_file_list:
                    print(img_name + "not in dir")
                    continue
                new_img_name = sample + img_name
                image_list.append(new_img_name)
                xmin_list.append(x1)
                ymin_list.append(y1)
                xmax_list.append(x1 + w)
                ymax_list.append(y1 + h)
                class_idx = int(data[1])
                classe = list(classes.keys())[class_idx]
                label_list.append(classe)
                copyfile(complete_img_name, dest_path + new_img_name)

        annotations_temp = pd.DataFrame(data={'image': image_list,
                                              'xmin': xmin_list,
                                              'ymin': ymin_list,
                                              'xmax': xmax_list,
                                              'ymax': ymax_list,
                                              'label': label_list})
        annotations = pd.concat([annotations, annotations_temp])

    print(str(annotations.shape[0]) + " images")
    annotations.to_csv(dest_path + "Annotations.csv", sep=",", index=False)

    zip_dir(dest_folder)
