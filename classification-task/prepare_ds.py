import pandas as pd
import numpy as np
import os
import shutil

DATA_DIR = 'data/images/'
DEST_DIR_TRAIN = 'data/train/'
DEST_DIR_TEST = 'data/test/'
METADATA = 'data/metadata.csv'

test_split = 0.1

metadata = pd.read_csv(METADATA)

# Get class labels
classes = metadata['dx']
classes = np.unique(classes)

if not os.path.exists(DEST_DIR_TRAIN):
    os.makedirs(DEST_DIR_TRAIN)

if not os.path.exists(DEST_DIR_TEST):
    os.makedirs(DEST_DIR_TEST)

for c in classes:
    # Create Directories for classes
    os.mkdir(DEST_DIR_TRAIN + str(c) + "/")
    os.mkdir(DEST_DIR_TEST + str(c) + "/")

    # Filter for label
    samples = metadata[metadata['dx'] == c]

    # Filter duplicates
    samples = samples.drop_duplicates(subset=['lesion_id'])

    # Shuffle image ids
    samples = samples['image_id']
    samples = samples.sample(frac = 1)

    # Divide into train and test set
    sample_test_length = int(len(samples) * test_split)
    samples_train = samples[:len(samples) - sample_test_length]
    samples_test = samples[len(samples) - sample_test_length:]

    for id in samples_train:
        shutil.copyfile((DATA_DIR + id +".jpg"), (DEST_DIR_TRAIN + c + "/"+id+".jpg"))

    for id in samples_test:
        shutil.copyfile((DATA_DIR + id +".jpg"), (DEST_DIR_TEST + c + "/"+id+".jpg"))



    

