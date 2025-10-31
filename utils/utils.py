from config.config import Config
import numpy as np


def load_config():
    return Config()

def generate_confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }