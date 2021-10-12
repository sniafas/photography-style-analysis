[![python-version](https://img.shields.io/badge/python-v3.8-blue&?style=flat)](https://www.python.org/) 
![docker](https://shields.io/badge/tensorflow-2.4-simple?logo=tensorflow&style=flat)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Photography Style Analysis Using Machine Learning

## Download images

Lite dataset contains 25K images in high resolution [~211GB].
For this project, images are transformed and released in

- **Horizontal (900x600) dataset:** [⬇️images](https://drive.google.com/file/d/1zgNqZMDsziN0M3gCeXSHOkmP6wW_owsG/view?usp=sharing) [2.21GB]

- **Square (400x400) dataset:** [⬇️images](https://drive.google.com/file/d/1E6TqDAy9BBc0rP2WN92F8pUmA5x3LfQt/view?usp=sharing) [1.29GB]


For more details about images, please check CHANGELOG->DATA and DOCS->image_analysis

# Repository organization

## Running scripts



## Image transformation

From im_analysis module
```
python im_analysis.py
```

## Training

To initiate training, you will need a CUDA capable GPU with cuda/cuDNN drivers enabled.
1. Edit ``configuration.json``
2. (optional) Edit architecture from ``architectures`` module
2. Train model

```
python train_model.py
```