import logging
import numpy as np
import torch
from sklearn import metrics


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()


def bootstrap_regression(x_pred, x_true, x_mask):
    N = len(x_true)
    N_idx = np.arange(N)
    K = 1000
    mse = []
    mae = []
    for _ in range(K):
        boot_idx = np.random.choice(N_idx, N, replace=True)
        boot_true = x_true[boot_idx]
        boot_pred = x_pred[boot_idx]
        boot_mask = x_mask[boot_idx]
        mse.append(mean_squared_error(boot_true, boot_pred, boot_mask).cpu())
        mae.append(mean_absolute_error(boot_true, boot_pred, boot_mask).cpu())
    return mse, mae


def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose(
            (1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        logging.info("confusion matrix:")
        logging.info(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls,
     thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score = 2 * prec1 * rec1 / (prec1 + rec1)
    if verbose:
        logging.info("accuracy = {}".format(acc))
        logging.info("precision class 0 = {}".format(prec0))
        logging.info("precision class 1 = {}".format(prec1))
        logging.info("recall class 0 = {}".format(rec0))
        logging.info("recall class 1 = {}".format(rec1))
        logging.info("AUC of ROC = {}".format(auroc))
        logging.info("AUC of PRC = {}".format(auprc))
        logging.info("min(+P, Se) = {}".format(minpse))
        logging.info("f1_score = {}".format(f1_score))

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse,
        "f1_score": f1_score
    }


def bootstrap(y_pred, y_true):
    N = len(y_true)
    N_idx = np.arange(N)
    K = 1000
    auroc = []
    auprc = []
    minpse = []
    acc = []
    f1_score = []
    y_pred = np.array(y_pred)
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    y_true = np.array(y_true)
    for _ in range(K):
        boot_idx = np.random.choice(N_idx, N, replace=True)
        boot_true = y_true[boot_idx]
        boot_pred = y_pred[boot_idx, :]
        ret = print_metrics_binary(boot_true, boot_pred, verbose=0)
        auroc.append(ret['auroc'])
        auprc.append(ret['auprc'])
        minpse.append(ret['minpse'])
        acc.append(ret['acc'])
        f1_score.append(ret['f1_score'])
    return auroc, auprc, minpse, acc, f1_score
