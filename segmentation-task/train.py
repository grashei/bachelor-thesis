import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        ReduceLROnPlateau,
                                        EarlyStopping)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from utils.create_dataset import create_dataset
import segmentation_models as sm
from model.mobunet import MobUnet

BUFFER_SIZE = 100
AUTOTUNE = tf.data.AUTOTUNE


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    batch_size = args.batch_size
    aug_num = args.augs
    epochs = args.epochs

    train = create_dataset(tf.estimator.ModeKeys.TRAIN,
                           aug=True,
                           aug_num=aug_num)
    test = create_dataset(tf.estimator.ModeKeys.EVAL)

    train_dataset = train.cache() \
        .shuffle(BUFFER_SIZE) \
        .batch(batch_size) \
        .prefetch(buffer_size=AUTOTUNE)

    test_dataset = test.cache() \
        .batch(batch_size) \
        .prefetch(buffer_size=AUTOTUNE)

    if args.model == 'mobilenetv2':
        model = MobUnet(dropout=args.dropout)
        
        model = model.call()
    else:
        model = sm.Unet(args.model, classes=2, encoder_weights='imagenet')

    loss = SparseCategoricalCrossentropy(from_logits=True)
    opt = Adam(args.lr)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])
    model.summary()

    checkpoint_filepath = ('{0}/{1}-unet_{2}'
                           '_{3}_{4}') \
        .format(args.model_path, args.model, args.augs,
                args.lr, args.dropout)
    print('\nModel will be saved to: ', checkpoint_filepath)

    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_weights_only=False,
                                       monitor='val_accuracy',
                                       mode='max',
                                       save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  min_lr=1e-6,
                                  verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   patience=6)

    model.fit(train_dataset, epochs=epochs,
              validation_data=test_dataset,
              callbacks=[early_stopping, model_checkpoint, reduce_lr])
    
    print('\ntraining done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        dest="model",
                        type=str,
                        choices=['densenet169', 'densenet201', 'efficientnetb0', 'efficientnetb5', 'inceptionv3', 'mobilenetv2', 'resnet101'],
                        required=False,
                        default='mobilenetv2',
                        help='Backbone to use')
    parser.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=100,
                        help='Number of training epochs')
    parser.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=0.001,
                        help='Learning rate to use')
    parser.add_argument("-aug", "--augs",
                        dest="augs",
                        type=int,
                        required=False,
                        default=7,
                        help="""Number of augmentations:
                        1. flip_left_right,
                        2. random_brightness,
                        3. adjust_saturation,
                        4. flip_up_down,
                        5. random_contrast,
                        6. random_brightness + adjust_saturation,
                        7. cutout,
                        8. rotate90,
                        9. adjust_gamma (0.5),
                        10. adjust_gamma (0.8),
                        """)
    parser.add_argument("-d", "--dropout",
                        dest="dropout",
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout to use. If 0.0 -> no dropout')
    args = parser.parse_args()
    print('\nTraining with args:\n', args)
    main(args)
