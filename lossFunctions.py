from re import M
from networkx import adjacency_data
import torch
import torch.nn as nn
import numpy as np
from itertools import compress
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, accuracy_score
from Arguments import argparser

ARGS = argparser()

def ReconAtomACC(reconAtom, gtAtom, mask):
    mask = mask.view(-1)
    reconAtom = reconAtom.view(-1, 119)[mask]
    reconAtom = torch.argmax(reconAtom, dim = 1)
    reconAtomNp = reconAtom.cpu().numpy()
    gtAtom = gtAtom.view(-1)[mask].cpu().numpy()
    atomACC = accuracy_score(gtAtom, reconAtomNp)
    return atomACC

def ReconAdjACC(reconAdj, gtAdj):
    reconAdj = reconAdj.view(-1)
    reconAdj = (reconAdj > 0.5).cpu().numpy()
    gtAdj = gtAdj.view(-1).cpu().numpy()
    adjAcc = roc_auc_score(gtAdj, reconAdj)
    return adjAcc

def ComputeKLDLoss(mu, var):
    KLDLoss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), 1)
    return torch.mean(KLDLoss)

def RegressionLoss(prey, y):
    mseLossFunction = nn.MSELoss()
    mseLoss = mseLossFunction(prey, y)
    return mseLoss

def MAELoss(prey, y):
    maeLossFunction = nn.L1Loss()
    maeLoss = maeLossFunction(prey, y)
    return maeLoss

def SingleClassificationLoss(prey, y):
    lossFn = nn.CrossEntropyLoss()
    loss = lossFn(prey, y)
    return loss

def MultiClassificationLoss(prey, y, labelMask):
    BCELossFunc = nn.BCEWithLogitsLoss(reduction="none")
    maskedLoss = BCELossFunc(prey, y) * labelMask
    loss = maskedLoss.sum() / labelMask.sum()
    return loss

def ComputeAUCScore(prey, y):
    predictions = np.array(prey)
    gt = np.array(y)
    aucSocre = roc_auc_score(gt, predictions[:,1])
    return aucSocre

def ComputeAUCScoreWithMultiClass(preds, labels, valid):
    """compute ROC-AUC and averaged across tasks"""
    preds = np.array(preds)
    labels = np.array(labels)
    valid = np.array(valid)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)

    rocauc_list = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")
    return sum(rocauc_list)/len(rocauc_list)

def ComputeRMSEScore(prey, y):
    mse = mean_squared_error(y, prey)
    rmse = np.sqrt(mse)
    return rmse

def ComputeMAEScore(prey, y):
    mae = mean_absolute_error(y, prey)
    return mae