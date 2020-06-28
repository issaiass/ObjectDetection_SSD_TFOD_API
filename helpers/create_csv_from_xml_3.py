import os
import fnmatch
import pandas as pd
import numpy as np
import shutil

from xml.etree import ElementTree as et

# Input Information
DATASET_FOLDER = 'Dataset'
TRAIN_XML = os.path.join(DATASET_FOLDER, 'Train')
TEST_XML  = os.path.join(DATASET_FOLDER, 'Test')

# Output Information
TRAIN_ANNOTATIONS = 'train_annotations.csv'
TEST_ANNOTATIONS  = 'test_annotations.csv'

print('[INFO] - Gathering XML Training Files')
xml_train_files = fnmatch.filter(os.listdir(TRAIN_XML), "*.xml")
xml_train_files_full_path = [os.path.join(TRAIN_XML, p) for p in xml_train_files]

print('[INFO] - Gathering XML Testing Files')
xml_test_files  = fnmatch.filter(os.listdir(TEST_XML), "*.xml")
xml_test_files_full_path = [os.path.join(TEST_XML, p) for p in xml_test_files]

print('[INFO] - Make the array of XML Files')
xml_files_full_path = [xml_train_files_full_path, xml_test_files_full_path]


folder_array = ['Train', 'Test'] # depending on the OS, os.listdir() could arrange differently
for i, TrainTestSet in enumerate(folder_array):
    xmlpaths = xml_files_full_path[i]
    print(f'[INFO] - Fixing the paths of the {TrainTestSet} XML Files')
    for xmlpath in xmlpaths:
        tree = et.parse(xmlpath)
        root = tree.getroot()
        for fname in root.iter('filename'):
            name = fname.text
        for path in root.iter('path'):
            path.text = os.path.join(DATASET_FOLDER, folder_array[i], name)
        print(f'[INFO] - New path of {TrainTestSet} set is = {path.text}')
        tree.write(xmlpath)


values = [[],[]]
for i, TrainTestSet in enumerate(folder_array):
    xmlpaths = xml_files_full_path[i]
    print(f'[INFO] - Fixing the values of the {TrainTestSet} XML Files')
    for xmlpath in xmlpaths:
        tree = et.parse(xmlpath)
        root = tree.getroot()
        for path in root.iter('path'):
            imagepath = path.text
        for s in root.iter('size'):
            w = int(s.find('width').text)
            h = int(s.find('height').text)
        for b in root.findall('.//bndbox'):
            xmin = int(b.find('xmin').text)
            ymin = int(b.find('ymin').text)
            xmax = int(b.find('xmax').text)
            ymax = int(b.find('ymax').text)
        for class_ in root.findall('.//object'):
            class_name = class_.find('name').text
        print(f'[INFO] - Making the array for {imagepath}')
        values[i].append([imagepath, w, h, class_name, xmin, ymin, xmax, ymax])

print('[INFO] - Making the dataframe of train ant test files')
train_df = pd.DataFrame(values[0])
test_df  = pd.DataFrame(values[1])

print('[INFO] - Marking column names of dataframes')
column_names = ['filename','width','height', 'class','xmin','ymin', 'xmax', 'ymax']
train_df.columns = column_names
test_df.columns = column_names

print('[INFO] - Saving data to CSV Files')
train_df.to_csv(TRAIN_ANNOTATIONS, index=False, header=True)
test_df.to_csv(TEST_ANNOTATIONS, index=False, header=True)


ANNOTATIONS_FOLDER = 'annotations'

print('[INFO] - Check annotation folder generation')
if not os.path.exists(ANNOTATIONS_FOLDER):
    os.mkdir(ANNOTATIONS_FOLDER)
if not os.path.exists(TRAIN_ANNOTATIONS):
    print('[INFO] - Please generate *train* annotations or see the *annotations* folder')
else:
    train_file = os.path.join(ANNOTATIONS_FOLDER, TRAIN_ANNOTATIONS)
    shutil.move(TRAIN_ANNOTATIONS, train_file)
    print('[INFO] - Moved train_annotations')
    
if not os.path.exists(ANNOTATIONS_FOLDER):
    os.mkdir(ANNOTATIONS_FOLDER)
if not os.path.exists(TEST_ANNOTATIONS):
    print('[INFO] - Please generate *test* annotations or see the *annotations* folder')
else:
    test_file  = os.path.join(ANNOTATIONS_FOLDER, TEST_ANNOTATIONS) 
    shutil.move(TEST_ANNOTATIONS, test_file)
    print('[INFO] - Moved test_annotations')