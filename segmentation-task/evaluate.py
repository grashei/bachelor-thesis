import tensorflow as tf
from utils.create_dataset import create_dataset
import pandas as pd
import os
import argparse

from utils.create_dataset import create_dataset

DATA_DIR = "data/test"

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_mask(pred_mask, idx=0):
    """Postprocess predicted mask

    Args:
        pred_mask: predicted mask
        idx (int, optional): [description]. Defaults to 0.

    Returns:
        pred_mask: processed mask
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = pred_mask[idx]

    return pred_mask


def sign_test(args):
    """Save model scores for significance test

    Args:
        args: cmd args
    """
    test_ds = create_dataset(tf.estimator.ModeKeys.PREDICT,
                             batch=False)
    
    models = [x for x in os.listdir(args.models_path)]

    for mdl in models:
        model = tf.keras.models.load_model(f'{args.models_path}/{mdl}')

        iou_scores = []
        miou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

        for idx, sample in enumerate(test_ds):
            image, mask = sample
            image = tf.expand_dims(image, axis=0)
            mask = tf.expand_dims(mask, axis=0)
            pred_mask = model.predict(image)
            p_mask = create_mask(pred_mask)

            miou_metric.update_state(mask[0], p_mask)
            iou_score = miou_metric.result().numpy()
            iou_scores.append(iou_score)
            df = pd.DataFrame(iou_scores, columns=[mdl])
            df.to_csv(f'scores_{mdl}.csv', index=True)


def test_model(args):
    """Get mIoU score for selected model

    Args:
        args: cmd args
    """
    test_ds = create_dataset(tf.estimator.ModeKeys.PREDICT,
                             batch=False)
    
    model = tf.keras.models.load_model(args.model)

    model.summary()

    iou_scores = []
    miou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

    for idx, sample in enumerate(test_ds):
        image, mask = sample
        image = tf.expand_dims(image, axis=0)
        mask = tf.expand_dims(mask, axis=0)

        pred_mask = model.predict(image)

        p_mask = create_mask(pred_mask)

        miou_metric.update_state(mask[0], p_mask)
        iou_score = miou_metric.result().numpy()
        iou_scores.append(iou_score)

        if args.save:
            msk = tf.keras.preprocessing.image.array_to_img(p_mask)
            msk.save(f"pred_img/{idx}_{iou_score}_unet.jpg", "JPEG")

            gt_msk = tf.keras.preprocessing.image.array_to_img(mask[0])
            gt_msk.save(f"pred_img/{idx}_gt.jpg", "JPEG")

            img = tf.keras.preprocessing.image.array_to_img(image[0])
            img.save(f"pred_img/{idx}_lesion.jpg", "JPEG")

    iou_score = tf.reduce_mean(iou_scores)
    print('mIoU: ', iou_score.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-md", "--mode",
                        dest="mode",
                        type=str,
                        choices=['test_model', 'sign_test'],
                        required=False,
                        default='test_model',
                        help='Test mode: whether to get mIoU score for 1\
                            model or to get data for significance test')
    parser.add_argument("-m", "--model",
                        dest="model",
                        type=str,
                        required=False,
                        help='Path to model')
    parser.add_argument("-mp", "--models_path",
                        dest="models_path",
                        type=str,
                        required=False,
                        help='Path to all models for significance test')
    parser.add_argument("-s", "--save",
                        dest="save",
                        type=str2bool,
                        required=False,
                        default='n',
                        help='Whether to save predicted mask and\
                            images (only for test_model mode)')
    args = parser.parse_args()
    print('\nTesting with args:\n', args)

    if args.mode == 'test_model':
        if not args.model:
            raise argparse.ArgumentError('Model path not specified!')
        test_model(args)
    elif args.mode == 'sign_test':
        if not args.models_path:
            raise argparse.ArgumentError('Models path not specified!')
        sign_test(args)
    else:
        pass
