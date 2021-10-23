[![python-version](https://img.shields.io/badge/python-v3.8-blue&?style=flat)](https://www.python.org/) 
![docker](https://shields.io/badge/tensorflow-2.4-simple?logo=tensorflow&style=flat)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Photography Style Analysis Using Machine Learning

## Download images

Lite dataset contains 25K images in high resolution [~211GB].
For this project, images are transformed and released in

- **Horizontal (900x600) images:** [⬇️download](https://drive.google.com/file/d/1zgNqZMDsziN0M3gCeXSHOkmP6wW_owsG/view?usp=sharing) [2.21GB]

- **Square (400x400) images:** [⬇️download](https://drive.google.com/file/d/1E6TqDAy9BBc0rP2WN92F8pUmA5x3LfQt/view?usp=sharing) [1.29GB]

- **Annotations**: [⬇️download](https://drive.google.com/drive/folders/1_R5zhWI-ZMQ7al_Pb7m8GXGyjjBI92Yf?usp=sharing)

- **Pretrained Models**: [⬇️download](https://drive.google.com/drive/folders/1eEZiWqc_LJa12QMf9qKzKh8-vR0gAKM-?usp=sharing)

For more details about images, please check CHANGELOG->DATA and DOCS->image_analysis

# Repository organization

## Image transformation

```
(image_analysis module)$ python im_analysis.py
```

## Training

To initiate training, the following configurations should be available


1. Edit ``configuration.json``
```
"training_mode": "sqal/sqal_ceal/all/all_ceal"
"ceal": true/false
"model_arch": "densenet/vgg"
"optimizer": "adam/nadam/rms/adagrad/sgd/adadelta"
```
2. Train model
```
python train_model.py
```

**Note:** A CUDA capable GPU with cuda/cuDNN drivers enabled and >=4GB RAM is recommended. TF 2.4 is compatible w/  `Cuda 11.0` and `cudnn v8.0.5 (November 9th, 2020), for CUDA 11.0`.

NVIDIA's archive: <https://developer.nvidia.com/rdp/cudnn-archive>

## Active Learning strategies

Active Learning w/ Incremental Training Active Learning Loop
![Active Learning Loop](/docs/active_learning_strategies/al_all_ceal.png)

Active Learning w/ Single Quary Active Learning
![Active Learning Loop](/docs/active_learning_strategies/al_sqal_ceal.png)