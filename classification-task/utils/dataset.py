import numpy as np
import tensorflow as tf

import pathlib
import os

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 100

def get_label(file_path, data_dir):
    class_dir = pathlib.Path(data_dir)
    class_names = np.array(sorted([item.name for item in class_dir.glob('[!.]*')]))
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_img(img, img_size):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_size, img_size])

def process_path(file_path, data_dir, img_size, aug=False, aug_num=None):
    label = get_label(file_path, data_dir)
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)

    if aug:
        if aug_num == 0:
            img = tf.image.flip_left_right(img)
        elif aug_num == 1:
            img = tf.image.random_brightness(img, 0.2)
        elif aug_num == 2:
            img = tf.image.random_saturation(img, 0.75, 1.25)
        elif aug_num == 3:
            img = tf.image.flip_up_down(img)
        elif aug_num == 4:
            img = tf.image.random_contrast(img, 0.5, 1.5)
        elif aug_num == 5:
            img = tf.image.rot90(img)
        elif aug_num == 6:
            tf.image.random_hue(img, 0.1)
        elif aug_num == 7:
            img = tf.image.adjust_gamma(img, 0.8)     
    
    return img, label

def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=BUFFER_SIZE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def create_dataset(data_dir, img_size, batch_size=32, aug=False, aug_num=None):
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob("*/*.jpg")))
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/"*/*"), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    
    val_size = int(image_count * 0.1)
    train = list_ds.skip(val_size)
    val = list_ds.take(val_size)
    
    print("Training samples: ", tf.data.experimental.cardinality(train).numpy())
    print("Validation samples: ", tf.data.experimental.cardinality(val).numpy())
    
    train_ds = train.map(lambda x: process_path(x, data_dir, img_size), num_parallel_calls=AUTOTUNE)
    
    if aug:
        for i in range(aug_num):
            aug_dataset = train.map(lambda x: process_path(x, data_dir, img_size, aug=aug, aug_num=i), num_parallel_calls=AUTOTUNE)
            train_ds = train_ds.concatenate(aug_dataset)

    val_ds = val.map(lambda x: process_path(x, data_dir, img_size), num_parallel_calls=AUTOTUNE)
    
    train_ds = configure_for_performance(train_ds, batch_size)
    val_ds = configure_for_performance(val_ds, batch_size)
    return train_ds, val_ds

def create_eval_dataset(data_dir, img_size, batch_size=32):
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    
    print('Test samples: ', tf.data.experimental.cardinality(list_ds).numpy())
    
    test_ds = list_ds.map(lambda x: process_path(x, data_dir, img_size), num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(test_ds, batch_size)
    return test_ds