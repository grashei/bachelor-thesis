import numpy as np
import tensorflow as tf

import sys
import pathlib
import os
import time

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
VAL_SPLIT = 0.25


@tf.function
def cutout(img, crop_size=10, crop_value=0., n_holes=2):
    h, w = img.shape[:-1]

    mask = np.ones((h, w))
    max_h = h - crop_size
    max_w = w - crop_size

    for n in range(n_holes):
        x = np.random.randint(0, max_h)
        y = np.random.randint(0, max_w)

        y1 = np.clip(y - crop_size // 2, 0, h)
        y2 = np.clip(y + crop_size // 2, 0, h)
        x1 = np.clip(x - crop_size // 2, 0, w)
        x2 = np.clip(x + crop_size // 2, 0, w)

        mask[y1: y2, x1: x2] = crop_value

    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    img = img * mask
    img = tf.convert_to_tensor(img)
    return img


def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    if batch_size > 0:
        ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def read_mask(fname, img_size):
    mask = tf.io.read_file(fname)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(images=mask, size=(img_size, img_size))
    thresh = 127
    mask = tf.cast(mask > thresh, tf.float32)
    return mask


def decode_img(img, img_size):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(images=img, size=(img_size, img_size))
    return img


def process_path(image_p, mask_p, data_dir, img_size, aug=False, aug_num=None):
    img = decode_img(image_p, img_size)
    mask = read_mask(mask_p, img_size)
    
    if aug:
        if aug_num == 0:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        elif aug_num == 1:
            img = tf.image.random_brightness(img, max_delta=0.5)
        elif aug_num == 2:
            img = tf.image.adjust_saturation(img, 3)
        elif aug_num == 3:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        elif aug_num == 4:
            img = tf.image.random_contrast(img, 0.2, 0.5)
        elif aug_num == 5:
            img = tf.image.random_brightness(img, max_delta=0.4)
            img = tf.image.adjust_saturation(img, 3)
        elif aug_num == 6:
            img = cutout(img)
        elif aug_num == 7:
            img = tf.image.rot90(img)
            mask = tf.image.rot90(mask)
        elif aug_num == 8:
            img = tf.image.adjust_gamma(img, 0.5)
        elif aug_num == 9:
            img = tf.image.adjust_gamma(img, 0.8) 
    
    return img, mask


def create_source_dataset(data_dir, img_size, batch_size=32, aug=False, aug_num=None):
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob("images/*.jpg")))
    
    images_ds = tf.data.Dataset.list_files(str(data_dir/"images/*.jpg"), shuffle=False)
    masks_ds = tf.data.Dataset.list_files(str(data_dir/"masks/*.png"), shuffle=False)
    
    list_ds = tf.data.Dataset.zip((images_ds, masks_ds))
    list_ds = list_ds.shuffle(BUFFER_SIZE)
    
    val_size = int(image_count * VAL_SPLIT)
    train = list_ds.skip(val_size)
    val = list_ds.take(val_size)
    
    print("Training samples: ", tf.data.experimental.cardinality(train).numpy())
    print("Validation samples: ", tf.data.experimental.cardinality(val).numpy())
    
    train_ds = train.map(lambda img, mask: process_path(img, mask, data_dir, img_size))
    
    if aug:
        for i in range(aug_num):
            aug_dataset = train.map(lambda img, mask: process_path(img, mask, data_dir, img_size, aug=aug, aug_num=i), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.concatenate(aug_dataset)
    
    val_ds = val.map(lambda img, mask: process_path(img, mask, data_dir, img_size))
    
    train_ds = configure_for_performance(train_ds, batch_size)
    val_ds = configure_for_performance(val_ds, batch_size)
    return train_ds, val_ds

def create_eval_dataset(data_dir, img_size, batch_size=32):
    data_dir = pathlib.Path(data_dir)
    
    images_ds = tf.data.Dataset.list_files(str(data_dir/"images/*.jpg"), shuffle=False)
    masks_ds = tf.data.Dataset.list_files(str(data_dir/"masks/*.png"), shuffle=False)
    
    list_ds = tf.data.Dataset.zip((images_ds, masks_ds))
    
    print("Test samples: ", tf.data.experimental.cardinality(list_ds).numpy())
    
    test_ds = list_ds.map(lambda img, mask: process_path(img, mask, data_dir, img_size))
    test_ds = configure_for_performance(test_ds, batch_size)
    return test_ds
    
    
    