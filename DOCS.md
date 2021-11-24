# Dataset

## csv/dof

The DoF based annotated dataset

# Source

## architectures

Various deep learning architectures

## configuration

Project's configuration entry point. Configuration is used from modules of the project for more efficient control and scaling up

## computational_graph

Computational graph for passive and active learning

## image_analysis

Image analysis handles the image transformation before training process.
 - It can transform images to any given ratio with ratio handling for both dimensions. When transforming and image to new ratio, it might not exactly fit to the target ratio 
    - exceeding resolution -> crop image to fit ratio
    - missing resolution -> 0pad image to fit ratio
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

## transformations

Basic image and label transformations 

## utils

Module with various utilities for data exploration, downloading images, plotting and training

## train_model.py 

Entry point to start training
