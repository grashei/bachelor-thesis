import tensorflow as tf
import argparse

from utils.dataset_dann import create_source_dataset
from utils.dataset import create_dataset
from model.DANN import DANN, DomainClassifierCallback

BATCH_SIZE = 32
IMG_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 100
SOURCE_DATA_DIR = 'data/data'
TARGET_DATA_DIR = 'data/train'


def train_tf(train_ds, val_ds, model, lr, epochs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[DomainClassifierCallback(epochs)]
    )

def main(args):
    
    # Source dataset
    """source_ds = create_source_dataset(tf.estimator.ModeKeys.TRAIN) \
        .cache() \
        .shuffle(buffer_size=BUFFER_SIZE) \
        .batch(BATCH_SIZE, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)"""

    source_ds, val_ds = create_source_dataset(SOURCE_DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    

    # Target dataset
    target_ds, _ = create_dataset(
        TARGET_DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Length of source dataset and target dataset must be equal
    target_ds = target_ds.take(len(list(source_ds)))

    # Validation dataset
    """val_ds = create_source_dataset(tf.estimator.ModeKeys.EVAL) \
        .cache() \
        .batch(BATCH_SIZE, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)"""

    # Build domain adaption dataset
    s_images = [d for (d, l) in source_ds]
    s_labels = [l for (d, l) in source_ds]
    t_images = [d for (d, l) in target_ds]

    da_ds = tf.data.Dataset.from_tensor_slices((s_images, s_labels, t_images))
    da_ds = da_ds.cache()
    da_ds = da_ds.prefetch(buffer_size=AUTOTUNE)

    # Create u-net DANN model
    model = DANN()

    train_tf(da_ds, val_ds, model, args.lr, args.epochs)

    model.save(args.model_path + '/DANN')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=50,
                        help='Number of training epochs')
    parser.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=1e-05,
                        help='Learning rate to use')
    
    args = parser.parse_args()
    print('\nTraining with args:\n', args)
    main(args)

