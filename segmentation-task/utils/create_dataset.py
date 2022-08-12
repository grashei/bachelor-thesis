import tensorflow as tf
import numpy as np


input_shape = 224


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


def read_image(fname, mode):
    image = tf.io.read_file(fname)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(images=image, size=(input_shape, input_shape))

    return image


def read_mask(fname, mode):
    mask = tf.io.read_file(fname)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(images=mask, size=(input_shape, input_shape))
    thresh = 127
    mask = tf.cast(mask > thresh, tf.float32)

    return mask


def read_line(line, mode, aug=False, aug_num=None):
    res = tf.io.decode_csv(line, record_defaults=[[""], [""]], field_delim=',')
    img = read_image(res[0], mode)
    mask = read_mask(res[1], mode)

    if aug:
        # rand = np.random.randint(0, aug_num)
        rand = aug_num
        if rand == 0:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        elif rand == 1:
            img = tf.image.random_brightness(img, max_delta=0.5)
        elif rand == 2:
            img = tf.image.adjust_saturation(img, 3)
        elif rand == 3:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        elif rand == 4:
            img = tf.image.random_contrast(img, 0.2, 0.5)
        elif rand == 5:
            img = tf.image.random_brightness(img, max_delta=0.4)
            img = tf.image.adjust_saturation(img, 3)
        elif rand == 6:
            img = cutout(img)
        elif rand == 7:
            img = tf.image.rot90(img)
            mask = tf.image.rot90(mask)
        elif rand == 8:
            img = tf.image.adjust_gamma(img, 0.5)
        elif rand == 9:
            img = tf.image.adjust_gamma(img, 0.8)

    return img, mask


def create_dataset(mode=None,
                   batch=False,
                   batch_size=32,
                   aug=False,
                   aug_num=None):
    train_dataset = tf.data.TextLineDataset(
        ['training_ds/seg_train.txt'])
    val_dataset = tf.data.TextLineDataset(
        ['training_ds/seg_val.txt'])
    test_dataset = tf.data.TextLineDataset(
        ['training_ds/seg_test.txt'])
    buffer_size = 600
    if aug:
        buffer_size = 1000

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_ = train_dataset.shuffle(buffer_size)
        train_dataset = train_.map(lambda x: read_line(x, mode))
        if aug:
            for i in range(aug_num):
                aug_dataset = train_.map(lambda x: read_line(x, mode,
                                                             aug=aug,
                                                             aug_num=i))
                train_dataset = train_dataset.concatenate(aug_dataset)
        if batch:
            train_dataset = train_dataset.batch(batch_size)
        return train_dataset
    elif mode == tf.estimator.ModeKeys.EVAL:
        val_dataset = val_dataset.map(lambda x: read_line(x, mode))
        if batch:
            val_dataset = val_dataset.batch(batch_size)
        return val_dataset
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_dataset = test_dataset.map(lambda x: read_line(x, mode))
        if batch:
            test_dataset = test_dataset.batch(batch_size)
        return test_dataset
