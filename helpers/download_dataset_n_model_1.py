import os
import tarfile
import argparse
import six.moves.urllib as urllib
import zipfile
from urllib.request import urlretrieve


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, 
                help="Utility for model download")
args = vars(ap.parse_args())

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = args["model"]
#MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

CURRENT_PATH = os.getcwd()

# Download Model
print(f'[INFO] - Model name = {MODEL_NAME}')
print('[INFO] - Downloading model')
if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name or 'ckpt' in file_name:
            print(f"Extracting {file_name}")
            tar_file.extract(file, os.getcwd())
print(f'[INFO] - Finished downloading model')
os.rename(MODEL_NAME, 'model')

DOWNLOAD_BASE = 'https://github.com/issaiass/ObjectDetection_Retinanet/raw/master/'
DATASET_NAME  = 'Dataset'
DATASET_FILE  = DATASET_NAME + '.zip'

# Download Dataset
print(f'[INFO] - Dataset name = {DATASET_NAME}')
# Download Model
print('[INFO] - Downloading dataset')
if not os.path.exists(DATASET_FILE):
    urlretrieve(DOWNLOAD_BASE + DATASET_FILE, DATASET_FILE)
    zip_file = zipfile.ZipFile(DATASET_FILE, 'r')
    print('[INFO] - Decompressing Dataset')
    zip_file.extractall()
print(f'[INFO] - Finished downloading dataset')