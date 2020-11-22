import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from configuration.config import Configuration
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt

config = Configuration().get_configuration()

csv_path = config['tfrecords']['csv_data_path']
img_path = config['tfrecords']['img_path']
io_path = config['tfrecords']['io_path']

# Used only when reading tf records
LABEL_UNPACK = config['training']['label']

IMAGE_SIZE = [config['training']['img_size_y'],
              config['training']['img_size_x']]

#NORMALISE = config['training']['normalise']
LABEL_1H = config['training']['1h']

# Used only when writing tfrecords
IMAGES_PER_TF_RECORD = 250



def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    # return tf.train.Feature(int64_list=tf.train.Int64List(value=value[0]))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value[0]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_list_feature(value):
    """Wrapper for inserting bytes list features into Example proto."""
    #value = [x.encode('utf8') for x in value[0]]
    value = [x.encode('utf8') for x in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if not isinstance(value, bytes):
        value = value.encode('utf8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""

    if isinstance(text, str):
        return text
    else:
        return str(text)


def _tfrecord_parse_map_function(example_proto):
    """
    Parse the input tf.Example proto using the dictionary
    This method is used on the fly when reading a tfrecord
    """
    #'DoF, DoF_bin', 'focal_label', 'focal_label_bin', 'exposure_label', 'iso_noise_label','iso_noise_label_bin'

    img_desc = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'DoF': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'DoF_bin': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }

    example = tf.io.parse_example([example_proto], img_desc)

    # unpack tfrecord
    img = example['image/encoded'][0]
    dof = example[LABEL_UNPACK][0]
    filename = example['image/filename'][0]

    # Image Decode
    x = tf.image.decode_jpeg(img, channels=3)
    x = tf.cast(x, tf.float32) #tf.uint8)
    
    '''
    if NORMALISE:
        x = tf.reshape(x / 255, [*IMAGE_SIZE, 3])
    else:
        x = tf.reshape(x, [*IMAGE_SIZE, 3])
    '''

    # Label decode
    y = tf.cast(dof, tf.int32)
    if LABEL_1H:
        y = tf.one_hot(y, 2)

    return x, y#, filename


class TFRecordIO:
    """
    TFRecord input/output procedures

    Args:
        csv_path: Path to read .csv raw data
        img_path: Path to read images for .tfrecord encoding
        io_path: Path to write .tfrecords. Caution: it is a prefix
    """

    def __init__(self, csv_path=csv_path, img_path=img_path, io_path=io_path):

        assert os.path.exists(csv_path), "File {} does not exist".format(
            csv_path
        )
        assert os.path.exists(img_path), "Image path {} does not exist".format(
            img_path
        )
        assert os.path.exists(io_path), "Read path {} does not exist".format(
            io_path
        )

        self.csv_path = csv_path
        self.img_path = img_path
        self.io_path = io_path

    #def __call__(self, dataset_type):
    #    return self.read_tf_record(dataset_type)

    def split(self, df, dataset_type, group='photo_id'):
        '''
        Group dataset by photo ids
        '''
        data = namedtuple(dataset_type, ['filename', 'labels'])
        gb = df.groupby(group)

        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(self, group, img_path, dataset_type):
        """
        Create tf_example dtype 

        Args: 
            group: 
            Image path
        Return:
            tf.train.Example        
        """

        # Join img_path with image ids acquired from group iteration
        with tf.io.gfile.GFile(os.path.join(img_path, '{}.jpg'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        classes = []

        for _, label in group.labels.iterrows():

            # Test data contain NaN values, we have to replace them constants
            # otherwise tf.Example produces error
            label_data = []
            '''
            if dataset_type == 'test':

                if np.isnan(label['DoF']):
                    label_data.append(5)
                else:
                    label_data.append(int(label['DoF']))

                if np.isnan(label['focal_label']):
                    label_data.append(5)
                else:
                    label_data.append(int(label['focal_label']))

                if np.isnan(label['exposure_label']):
                    label_data.append(5)
                else:
                    label_data.append(int(label['exposure_label']))

                if np.isnan(label['iso_noise_label']):
                    label_data.append(5)
                else:
                    label_data.append(int(label['iso_noise_label']))

            else:
            '''
            # This array is case sensitive to the features names in csv dataset
            # and is used to create labels in tfrecords
            label_data = [
                int(label['DoF']),
                int(label['DoF_bin']),
                int(label['focal_label']),
                int(label['focal_label_bin']),
                int(label['exposure_label']),
                int(label['iso_noise_label']),
                int(label['iso_noise_bin_label'])
            ]
            classes.append(label_data)

        # tf example creates the tfrecord format for each entry in the record
        # names are not case sensitive and are custom names for that represent the
        # corresponded object in tfrecord
        # extra caution for the classes slicing, must follow the above structure
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/filename': _bytes_feature(filename),
            'image/encoded': _bytes_feature(encoded_jpg),
            'image/format': _bytes_feature(image_format),
            'DoF': _int64_feature(classes[0][0]),
            'DoF_bin': _int64_feature(classes[0][1]),
            'focal_label': _int64_feature(classes[0][2]),
            'focal_label_bin': _int64_feature(classes[0][3]),
            'exposure_label': _int64_feature(classes[0][4]),
            'iso_noise_label': _int64_feature(classes[0][5]),
            'iso_noise_label_bin': _int64_feature(classes[0][6]),
        }))

        return tf_example

    def write_tf_record(self, dataset_type, max_images_per_file=IMAGES_PER_TF_RECORD):
        """
        Args:
            dataset_type: Define that dataset type eg. (train, valid, test) to create .tfrecords
            max_images_per_file: img/label pairs per tf_record
        """

        filename_idx = 1
        input_data = glob(self.csv_path + dataset_type + '*')
        raw_data = pd.read_csv(input_data[0])
        grouped_data = self.split(raw_data, dataset_type)
        
        for idx, g in enumerate(grouped_data):
            #if idx % max_images_per_file == 0:
            #    print(idx)
            # Create .tfrecord frequency
            
            if idx % max_images_per_file == 0:
                writer = tf.io.TFRecordWriter(self.io_path +
                                              dataset_type +
                                              '-%02d.tfrecord' % (filename_idx)
                                              )
                print("Writing:...: " + dataset_type +
                      '-%02d.tfrecord' % (filename_idx))
                filename_idx += 1

            # Create tf record structure and populate with csv data
            tf_example = self.create_tf_example(g, self.img_path, dataset_type)

            # Serialize and write to file
            writer.write(tf_example.SerializeToString())
            

        writer.close()
        
    def read_tf_record(self, dataset_type):

        # Read paths of tf_records
        tf_records = tf.io.gfile.glob(self.io_path + dataset_type + '*')
        record = tf.data.TFRecordDataset(tf_records)
        # Diserialize tf records
        decoded_tf_record = record.map(
            _tfrecord_parse_map_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        tf_records_len = sum(1 for record in decoded_tf_record)
       
        return decoded_tf_record, tf_records_len
