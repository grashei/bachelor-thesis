import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.dataset import create_eval_dataset
from sklearn.metrics import (recall_score, confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score)

NUM_CLASSES = 7

data_dir_test = 'data/test'
label = [ 'akiec', 'bcc','bkl','df','mel', 'nv',  'vasc']

  
def main(args):
    model = tf.keras.models.load_model(args.model_path)
    model.summary()
    
    test_ds = create_eval_dataset(data_dir_test, img_size=args.size, batch_size=args.batch_size)
    
    results = model.evaluate(test_ds)
    print("test loss, test acc:", results)
    
    correct_classes = [0 for i in range(len(label))]
    total_classes = [0 for i in range(len(label))]
    sum = 0
    all_predictions = []
    all_labels = []

    for b in test_ds:
        images, labels = b
        predictions = model.predict(images)
        predicted_labels = tf.math.argmax(predictions, axis=1).numpy()
        
        all_predictions.append(predicted_labels)
        all_labels.append(labels)
        
        for l, p in zip(labels, predicted_labels):
            sum += 1
            total_classes[l] += 1
            
            if l == p:
                correct_classes[l] += 1
                
    print('Total samples correct: ',correct_classes)
    print('Total samples: ', total_classes)
    print('Sum samples: ', sum)
    
    y_pred = [item for sublist in all_predictions for item in sublist]
    y_true = [item for sublist in all_labels for item in sublist]
    
    for idx, l in enumerate(label):
        recall = recall_score(y_true, y_pred, average=None, labels=[idx])
        print('Recall for ', l, ': ', recall)
        
    print('Unweighted recall: ', balanced_accuracy_score(y_true, y_pred))
    
        
    confusion_matrix_tensorflow = tf.math.confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:\n', confusion_matrix_tensorflow)
    
    cm = confusion_matrix(y_true,y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('conf_matrix.svg', format='svg')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path",
                        dest="model_path",
                        type=str,
                        required=True,
                        help='Path to model')
    parser.add_argument("-bs", "--batch_size",
                        dest="batch_size",
                        type=int,
                        required=False,
                        default=32,
                        help='Batch size')
    parser.add_argument("-s", "--size",
                        dest="size",
                        type=int,
                        required=False,
                        default=224,
                        help='Input image size')

    args = parser.parse_args()
    print('\nEvaluating with args:\n', args)
    main(args)