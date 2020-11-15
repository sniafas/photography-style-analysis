# Msc-Thesis

Repository of my Msc Data Science thesis. 

## Download images

Lite dataset contains 25K images in high resolution [~211GB].
For this project, images are transformed and released in

- **Square (400x400) dataset:** [⬇️images](https://drive.google.com/file/d/1Eht4PDWlRWhWalWXP53da8CvaWy1TBEx/view?usp=sharing) [1.4GB], [⬇️tfrecords](https://drive.google.com/file/d/1kTv-ANmbjIt20hKehGNOJohtS5HFUHzx/view?usp=sharing) [1.2GB]


- **Horizontal (300x200) dataset:** [⬇️images](https://drive.google.com/file/d/1h_Q5NvDRhaP0cNyjmwOg7wucCtBijy-0/view?usp=sharing) [418MB], [⬇️tfrecords](https://drive.google.com/file/d/1uGljFTsXoi-MMzxKjtirs5FeSmYAdDUR/view?usp=sharing) [414MB]

For more details about images, please check CHANGELOG->DATA and DOCS->image_analysis

# Repository organization

## Running scripts

Source code in the repository is organized in such way that training must be run from the root of the repository. Anything else can be launched from the ``interfaces`` module, where is stated.
If you use and IDE, set the *Working Directory* point to the root of the repository.

## Image transformation

From im_analysis module
```
python im_analysis.py
```

## Generate TFrecords

1. Edit ``configuration.json``
2. From ``interfaces`` module:
```
python write_tf_records.py
```

## Training

To run training, you will need a CUDA capable GPU with cuda/cuDNN drivers.
1. Edit ``configuration.json``
2. (optional) Edit architecture from ``architectures`` module
2. Train model

```
python train_model.py
```

## Dataset exploration - import new dataset

Launch corresponded *ipynb* notebooks from ``interfaces`` module
