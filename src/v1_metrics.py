import numpy as np
import logging
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, confusion_matrix

# EER reference: https://yangcha.github.io/EER-ROC/

def compute_eer(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1)
    
    return 100. * eer

def compute_confuse(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
