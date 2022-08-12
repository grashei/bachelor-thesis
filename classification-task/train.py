import argparse
import tensorflow as tf
import pandas as pd
import datetime

from utils.dataset import create_dataset
from utils.weighting import weights_sklearn
from model.model import build_model, build_unet_model, build_concatenated_model, unfreeze_model
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping)

data_dir = 'data/train'
metadata = pd.read_csv('data/metadata.csv')

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Functions #

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_model(train_ds, val_ds, model, lr, epochs):    
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_weights_only=False,
                                       monitor='val_accuracy',
                                       mode='max',
                                       save_best_only=True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=4,
                                  min_lr=1e-6,
                                  verbose=1)
    
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   verbose=1,
                                   patience=5,
                                   restore_best_weights=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    model.summary()
    
    classweights = weights_sklearn(metadata)
    
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=classweights,
        callbacks=[model_checkpoint, reduce_lr, early_stopping, tensorboard_callback])

    
def main(args):
    global checkpoint_filepath

    train_ds, val_ds = create_dataset(
        data_dir,
        img_size=args.size,
        batch_size=args.batch_size,
        aug=args.augs > 0,
        aug_num=args.augs
    )
    
    if args.cmd == "no_finetuning":
        checkpoint_filepath = ('{0}/{1}-{2}_{3}_{4}_{5}') \
            .format(args.model_path, args.cmd, args.model, args.augs, args.lr, args.dropout)
        model = build_model(args.model, args.dropout, img_size=args.size)
        train_model(train_ds, val_ds, model, args.lr, args.epochs)
        
    elif args.cmd == "finetuning":
        checkpoint_filepath = ('{0}/{1}-{2}_{3}_{4}_{5}_{6}') \
            .format(args.model_path, args.cmd, args.model, args.augs, args.lr, args.flr, args.dropout)
        model = build_model(args.model, args.dropout, img_size=args.size)
        train_model(train_ds, val_ds, model, args.lr, args.epochs)
        model.trainable = True
        train_model(train_ds, val_ds, model, args.flr, args.epochs)
        
    elif args.cmd == "blocks":
        checkpoint_filepath = ('{0}/{1}-{2}_{3}_{4}_{5}_{6}') \
            .format(args.model_path, args.cmd, args.augs, args.lr, args.flr, args.dropout, args.blocks)
        building_blocks = [12, 21, 30, 39, 48, 57, 65, 74, 83, 92, 101, 110, 119, 128, 137, 155]
        model = build_model('mobilenetv2', args.dropout, img_size=args.size)
        train_model(train_ds, val_ds, model, args.lr, args.epochs)
        model.trainable = True
        backbone = model.get_layer(name='mobilenetv2_1.00_224')
        unfreeze_model(backbone, building_blocks[args.blocks-1])
        train_model(train_ds, val_ds, model, args.flr, args.epochs)
            
    elif args.cmd == "unet":
        checkpoint_filepath = ('{0}/{1}-{2}_{3}_{4}_{5}') \
            .format(args.model_path, args.cmd, args.augs, args.lr, args.flr, args.dropout)
        model = build_unet_model(args.dropout, args.unet)
        train_model(train_ds, val_ds, model, args.lr, args.epochs)
        model.trainable = True
        train_model(train_ds, val_ds, model, args.flr, args.epochs)
        
    elif args.cmd == "concatenated":
        checkpoint_filepath = ('{0}/{1}-{2}_{3}_{4}') \
            .format(args.model_path, args.cmd, args.augs, args.flr, args.dropout)
        model = build_concatenated_model(args.dropout, args.unet)
        train_model(train_ds, val_ds, model, args.flr, args.epochs)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Select experiment", dest="cmd")

    parser_no_ft = subparsers.add_parser("no_finetuning", help="Run training without finetuning")
    parser_no_ft.add_argument("-m", "--model",
                        dest="model",
                        type=str,
                        choices=['densenet201', 'efficientnetb0', 'efficientnetb5', 'inceptionv3', 'mobilenetv2', 'mobilenetv3_l', 'nasnet', 'resnet', 'vgg19'],
                        required=False,
                        default='mobilenetv2',
                        help='Backbone to use')
    parser_no_ft.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser_no_ft.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser_no_ft.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=100,
                        help='Number of training epochs')
    parser_no_ft.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=0.0001,
                        help='Learning rate to use')
    parser_no_ft.add_argument("-aug", "--augs",
                        dest="augs",
                        type=int,
                        required=False,
                        default=7,
                        help="""Number of additive augmentations:
                        1. flip_left_right,
                        2. random_brightness,
                        3. adjust_saturation,
                        4. flip_up_down,
                        5. random_contrast,
                        6. rotate90,
                        7. adjust_gamma,
                        """)
    parser_no_ft.add_argument("-d", "--dropout",
                        dest="dropout",
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout to use. If 0.0 -> no dropout')
    parser_no_ft.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=300,
                        help='Image resizing size')


    parser_ft = subparsers.add_parser("finetuning", help="Run training with finetuning")
    parser_ft.add_argument("-m", "--model",
                        dest="model",
                        type=str,
                        choices=['densenet201', 'efficientnetb0', 'efficientnetb5', 'inceptionv3', 'mobilenetv2', 'mobilenetv3_l', 'nasnet', 'resnet', 'vgg19'],
                        required=False,
                        default='mobilenetv2',
                        help='Backbone to use')
    parser_ft.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser_ft.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser_ft.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=100,
                        help='Number of training epochs')
    parser_ft.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=0.0001,
                        help='Learning rate to use')
    parser_ft.add_argument("-flr", "--finetuning_lr",
                        dest="flr",
                        type=float,
                        required=False,
                        default=1e-5,
                        help='Finetuning learning rate')
    parser_ft.add_argument("-aug", "--augs",
                        dest="augs",
                        type=int,
                        required=False,
                        default=7,
                        help="""Number of additive augmentations:
                        1. flip_left_right,
                        2. random_brightness,
                        3. adjust_saturation,
                        4. flip_up_down,
                        5. random_contrast,
                        6. rotate90,
                        7. adjust_gamma (0.5),
                        """)
    parser_ft.add_argument("-d", "--dropout",
                        dest="dropout",
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout to use. If 0.0 -> no dropout')
    parser_ft.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=300,
                        help='Image resizing size')


    parser_blocks = subparsers.add_parser("blocks", help="Train MobileNet with given number of trainable building blocks")
    parser_blocks.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser_blocks.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser_blocks.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=100,
                        help='Number of training epochs')
    parser_blocks.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=0.0001,
                        help='Learning rate to use')
    parser_blocks.add_argument("-flr", "--finetuning_lr",
                        dest="flr",
                        type=float,
                        required=False,
                        default=1e-5,
                        help='Finetuning learning rate')
    parser_blocks.add_argument("-aug", "--augs",
                        dest="augs",
                        type=int,
                        required=False,
                        default=7,
                        help="""Number of additive augmentations:
                        1. flip_left_right,
                        2. random_brightness,
                        3. adjust_saturation,
                        4. flip_up_down,
                        5. random_contrast,
                        6. rotate90,
                        7. adjust_gamma (0.5),
                        """)
    parser_blocks.add_argument("-d", "--dropout",
                        dest="dropout",
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout to use. If 0.0 -> no dropout')
    parser_blocks.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=300,
                        help='Image resizing size')
    parser_blocks.add_argument("-b", "--blocks",
                        dest="blocks",
                        type=int,
                        required=False,
                        default=0,
                        help='Building blocks with finetuning')

    parser_unet = subparsers.add_parser("unet", help="Use MobileNet u-net encoder as feature extractor")
    parser_unet.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser_unet.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser_unet.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=100,
                        help='Number of training epochs')
    parser_unet.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=0.0001,
                        help='Learning rate to use')
    parser_unet.add_argument("-flr", "--finetuning_lr",
                        dest="flr",
                        type=float,
                        required=False,
                        default=1e-5,
                        help='Finetuning learning rate')
    parser_unet.add_argument("-aug", "--augs",
                        dest="augs",
                        type=int,
                        required=False,
                        default=7,
                        help="""Number of additive augmentations:
                        1. flip_left_right,
                        2. random_brightness,
                        3. adjust_saturation,
                        4. flip_up_down,
                        5. random_contrast,
                        6. rotate90,
                        7. adjust_gamma (0.5),
                        """)
    parser_unet.add_argument("-d", "--dropout",
                        dest="dropout",
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout to use. If 0.0 -> no dropout')
    parser_unet.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=224,
                        help='Image resizing size')
    parser_unet.add_argument("-unet",
                        dest="unet",
                        type=str,
                        required=True,
                        help='Path to unet model')

    parser_concat = subparsers.add_parser("concatenated", help="Use MobileNet u-net encoder and ImageNet encoder concatenated")
    parser_concat.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=False,
                        default='models',
                        help='Where to store checkpoints')
    parser_concat.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser_concat.add_argument("-e", "--epochs",
                        dest="epochs",
                        type=int,
                        required=False,
                        default=100,
                        help='Number of training epochs')
    parser_concat.add_argument("-lr", "--learning_rate",
                        dest="lr",
                        type=float,
                        required=False,
                        default=0.0001,
                        help='Learning rate to use')
    parser_concat.add_argument("-flr", "--finetuning_lr",
                        dest="flr",
                        type=float,
                        required=False,
                        default=1e-5,
                        help='Finetuning learning rate')
    parser_concat.add_argument("-aug", "--augs",
                        dest="augs",
                        type=int,
                        required=False,
                        default=7,
                        help="""Number of additive augmentations:
                        1. flip_left_right,
                        2. random_brightness,
                        3. adjust_saturation,
                        4. flip_up_down,
                        5. random_contrast,
                        6. rotate90,
                        7. adjust_gamma (0.5),
                        """)
    parser_concat.add_argument("-d", "--dropout",
                        dest="dropout",
                        type=float,
                        required=False,
                        default=0.2,
                        help='Dropout to use. If 0.0 -> no dropout')
    parser_concat.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=224,
                        help='Image resizing size')
    parser_concat.add_argument("-unet",
                        dest="unet",
                        type=str,
                        required=True,
                        help='Path to unet model')

    args = parser.parse_args()
    print('\nTraining with args:\n', args)
    main(args)