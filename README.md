# Bachelor Thesis Skin Lesion Segmentation and Classification with ISIC Challenge 2018 Dataset

This repository contains the python code used for my bachelor's thesis "Automatic Skin Lesion Segmentation and Classification Utilising Deep CNN-Based Transfer Learning". How the described models can be trained is stated below.

## Segmentation Task

1. Download data for the ISIC Challenge 2018 part 1 from [ISIC](https://challenge.isic-archive.com/data).
2. Use the data to create this directory structure inside the segmentation-task directory:
```
    data
    ├── images                  # Input images
    └── masks                   # Ground truth segmentation masks
```  
3. Prepare dataset by splitting into train, validation and test with `python prepare_ds.py`.
4. Train the CNN with `python train.py`. Hyperparamters can be set by using CLI arguments. The help can be shown by adding the argument `-h`.
5. Evaluate the trained model by executing `python test.py`. The help can be shown by adding the argument `-h`.

## Classification Task

1. Download data and ground truth for the ISIC Challenge 2018 part 3 from [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
2. Use the data to create this directory structure inside the classification-task directory:
```
    data
    ├── images                # All input images
    └── metadata.csv          # Ground truth csv
```  
3. Prepare dataset by executing `python prepare_ds.py`.
4.  Train the CNN with `python train.py` The CLI contains subcommands for all experiments:
```
positional arguments:
  {no_finetuning,finetuning,blocks,unet,concatenated}
                        Select experiment
    no_finetuning       Run training without finetuning
    finetuning          Run training with finetuning
    blocks              Train MobileNet with given number of trainable
                        building blocks
    unet                Use MobileNet u-net encoder as feature extractor
    concatenated        Use MobileNet u-net encoder and ImageNet encoder
                        concatenated

optional arguments:
  -h, --help            show this help message and exit
```
Hyperparamters for each experiment can be set by using CLI arguments after the subcommand. For each subcommand the help can be shown by adding the argument `-h`.


5. Evaluate the trained model by executing `python evaluate.py`. The help can be shown by adding the argument `-h`.
6. For training the DANN use `python train_DANN.py`. A symbolic link must be created to the segmentation data inside the data directory.

