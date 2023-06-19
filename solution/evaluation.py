import numpy as np


def get_tpr(preds, labels):
    """Calculates true positive rate"""

    # Number of true positives
    tp = np.sum(preds & labels)

    # Number of false negatives
    fn = np.sum(~preds & labels)

    return tp / (tp + fn)


def get_fpr(preds, labels):
    """Calculates false positive rate"""

    # Number of false positives
    fp = np.sum(preds & ~labels)

    # Number of true negatives
    tn = np.sum(~preds & ~labels)

    return fp / (fp + tn)
