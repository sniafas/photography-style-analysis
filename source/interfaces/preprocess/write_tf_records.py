"""
 To select the dataset of your choice please alter configuration.json

- .csv:
tfrecords -> csv_data_path

- image path:
tfrecords -> img_path

- path to write tfrecords:
tfrecords -> io_path

- tfrecords -> img_size_x, img_size_y
"""
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
from configuration.config import Configuration
from preprocess.tf_records import TFRecordIO
from utils.train_utils import load_dataset
import matplotlib.pyplot as plt

config = Configuration().get_configuration()

csv_path = config['tfrecords']['csv_data_path']
img_path = config['tfrecords']['img_path']
io_path = config['tfrecords']['io_path']
tfrio = TFRecordIO(csv_path, img_path, io_path)

# 'train', 'valid', 'test'
tfrio.write_tf_record('test')
