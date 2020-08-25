import numpy as np

def dice(targets, predictions):
    return (2*np.dot(predictions,targets)) / (predictions.sum() + targets.sum())