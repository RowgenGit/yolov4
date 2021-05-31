import pandas as pd
import os
import zipfile

def prepare_csv_file(path):
    images_and_boxes = pd.read_csv(path,sep=" ",names=["frame","id_class","id_annotation","x","y","w","h","smth"])
    test = pd.read_csv(path)
    print(test)
    print(images_and_boxes)
    return ""

def read_classes(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def get_file_path(dir):
    filepaths = []
    for root,directories,files in os.walk(dir):
        for filename in files:
            filepath =os.path.join(root,filename)
            filepaths.append(filepath)
    return filepaths


def zip_dir(dir):
    files = get_file_path(dir)
    zip_file = zipfile.ZipFile(dir + '.zip','w')
    with zip_file:
        #writing each file one by one
        for file in files:
            zip_file.write(file)


#training_data_path = "training_data/"
#csv_path = training_data_path + "Annotations.csv"
#prepare_csv_file(csv_path)

