# Dataset

## csv/dof

The DoF based annotated dataset

## csv/dataset.csv

The `dataset.csv` contains the entire dataset of 25K images. It contains all null and non null cleaned and free from noise EXIF features with photo orientation (1 horizontal, 0 vertical)

# Codebase

## architectures

Various architectures

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

## utils

Module with various utilities for data exploration, downloading images, plotting and training

## train_model.py 

Entry point to start training
