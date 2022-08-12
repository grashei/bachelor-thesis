import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

def weights_mfb(metadata):
    class_weights = {}
    labels = __get_labels__(metadata)
    counts = np.zeros_like(labels)
    for i,l in enumerate(labels):
        counts[i] = metadata[metadata['dx']==str(l)]['dx'].value_counts()[0]
    counts = counts.astype(np.float)
    median_freq = np.median(counts)
    for i, labels in enumerate(labels):
        class_weights[i] = median_freq / counts[i]
    return class_weights

def weights_proportional(metadata):
    class_weights = {}
    labels = __get_labels__(metadata)
    counts = np.zeros_like(labels)
    for i,l in enumerate(labels):
        counts[i] = metadata[metadata['dx']==str(l)]['dx'].value_counts()[0]
    counts = counts.astype(np.float)
    for i, labels in enumerate(labels):
        class_weights[i] = (1 / counts[i]) * (len(metadata) / 7)
    return class_weights

def weights_sklearn(metadata):
    classes = metadata['dx']
    classweights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    classweights = {i : classweights[i] for i in range(len(np.unique(classes)))}
    return classweights

def __get_labels__(metadata):
    classes = metadata['dx']
    labels = classes=np.unique(classes)
    return labels
    