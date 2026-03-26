import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)


def per_class_accuracy(conf_mat):
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        acc = np.nan_to_num(acc)
    return acc