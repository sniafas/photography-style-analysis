# Dataset

The Unsplash+ Dataset is a cleaned and labeled(ongoing process) version of unsplash dataset:

## dataset.csv

The `dataset.csv` contains the entire dataset of 25K image, one per row. It contains all null and non null cleaned and free from noise EXIF features with photo orientation (1 horizontal, 0 vertical)

## train - validation - test

`train.csv`, `validation.csv` and `test.csv` is a slice of the `dataset.csv` which contains samples only with non null EXIF (exif_iso, exif_focal_length, exif_aperture_values, exif_exposure_time) features along with experimental labeling.

| Field                       | Description |
|-----------------------------|-------------|
| photo_id                       | ID of the Unsplash photo |
| exif_camera_make               | Camera make (brand) extracted from the EXIF data |
| exif_camera_model              | Camera model extracted from the EXIF data |
| exif_iso                       | ISO setting of the camera, extracted from the EXIF data |
| exif_aperture_value            | Aperture setting of the camera, extracted from the EXIF data |
| exif_focal_length              | Focal length setting of the camera, extracted from the EXIF data |
| exif_exposure_time             | Exposure time setting of the camera, extracted from the EXIF data |
| orientation                    | 0: vertical, 1 horizontal |
| iso_noise_label                | 0: exif_iso  (0,500) , 1: exif_iso  [500,1800) | 2: exif_iso [1800,] 
| iso_noise_bin_label            | 0: exif_iso < 500, 1: exif_iso => 500 | 
| DoF_bin                        | 0: exif_aperture_value (0,3.5], 1: exif_aperture_value (3.5,]
| DoF                            | 0: exif_aperture_value (0,3.2), 1: exif_aperture_value [3.2,5.6], 2: exif_aperture_value (5.6,]|
| exposure_label                 | 0: exif_exposure_time > 0'', 1: exif_exposure_time <= 1/250, 2: exif_exposure_time > 1/250 |
| focal_label_bin                | 0: exif_focal_length (0, 85mm], 1: exif_focal_length (85mm,) |
| focal_label                    | 0: exif_focal_length (0, 35mm], 1: exif_focal_length (35mm,85mm], 2: exif_focal_length (85mm, ) |

# Source

```bash
.
├── 3rd party
│   └── unsplash
├── architectures
│   ├── __init__.py
│   └── model_architectures.py
├── configuration
│   ├── config.py
│   ├── configuration.json
│   └──__init__.py
├── image_analysis
│   ├── im_analysis.py
│   └── __init__.py
├── __init__.py
├── interfaces
│   ├── dataset_exploration.ipynb
│   ├── import_new_dataset.ipynb
│   ├── plt_style.py
│   ├── read_tf_records.ipynb
│   ├── resize2ratio.ipynb
│   └── write_tf_records.py
├── model
│   ├── inference.py
│   ├── __init__.py
│   └── training.py
├── preprocess
│   ├── __init__.py
│   └── tf_records.py
├── train_model.py
└── utils
    ├── data_utils.py
    ├── download.py
    ├── gpu_util.py
    ├── __init__.py
    ├── plot_utils.py
    ├── training_callbacks.py
    └── train_utils.py
```
## architectures

Training model architecture in Keras sequential

## configuration

Project's configuration entry point. Configuration is used from modules of the project for more efficient control and scaling up

## image_analysis

Image analysis handles the image transformation before tfrecord generation and training process.
 - It can transform images to any given ratio with ratio handling for both dimensions. When transforming and image to new ratio, it might not exactly fit to the target ratio 
    - exceeding resolution -> crop image to fit ratio
    - missing resolution -> 0pad image to fit ratio
- General resize images to reduce disk size(depricated, not used from the project), checks orientation(depricated, since new unsplash release),
Images have to follow a standard ratio (3:2).

## interfaces

Various interfaces in notebooks and .py executables to access project's modules. 
- Dataset exploration, dataset addition
    - Data Cleaning, wrangling
    - Image discrimination (Vertical-Horizontal) -> orientation.py (depricated)
    - New dataset accumulation
    - Dataset split (training, validation, test)
    - Exif analysis
    - Label distribution
    - Misc experimental section
- Read, write tfrecords
    - Show a tfrecord, plot image-label
    - Generate new tfrecords

## model

Training and inference module

## preprocess

TF records generation module

## utils

Module with various utilities for data exploration, downloading images, plotting and training

## train_model.py 

Entry point to start training

# TF records

TF records are small binary files that improve the performance substantially.
Project's tfrecords are generated with the following structure and naming(case_sensitive):
- image/filename
- image/encoded
- image/format
- DoF
- DoF_bin
- focal_label
- focal_label_bin
- exposure_label
- iso_noise_label
- iso_noise_label_bin
