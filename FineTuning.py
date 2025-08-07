

from tqdm import tqdm
from DataHelper import MolTuneDatasetWrapper
from torch.utils.data import random_split
import numpy as np
import torch
from VAE import VAE
from downstreamModel import DownstreamModel

import lossFunctions as LF
import sys
# from transformers import get_scheduler
from rdkit import RDLogger
import torch.nn.functional as F


# 禁用 RDKit 的日志
RDLogger.DisableLog('rdApp.*')

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def Train(taskType, model, device, dataLoader, optimizer, normalizer):
    KLDEpochLoss = []
    predictionEpochLoss = []
    totalEpochLoss = []
    
    model.train()
    with tqdm(dataLoader, disable = not sys.stdout.isatty()) as tqdmIter:
        for data, labelMask in tqdmIter:
            optimizer.zero_grad()
            data = data.cuda(device)
            labelMask = labelMask.to(device)
            prediction, molMu, molVar = model(data)

            molKLDLoss = LF.ComputeKLDLoss(molMu, molVar)
            if taskType == "classification":
                predictionLoss = LF.MultiClassificationLoss(prediction, data.y, labelMask)
            elif taskType == "RMSE":
                predictionLoss = LF.RegressionLoss(prediction, data.y)
            else:
                if normalizer:
                    predictionLoss = LF.MAELoss(prediction, normalizer.norm(data.y))
                else:
                    predictionLoss = LF.MAELoss(prediction, data.y)

            # loss = predictionLoss
            loss = predictionLoss + molKLDLoss * 1e-2
            loss.backward()
            optimizer.step()

            KLDEpochLoss.append(molKLDLoss.item())
            predictionEpochLoss.append(predictionLoss.item())
            totalEpochLoss.append(loss.item())
            tqdmIter.set_postfix(train_loss = np.mean(totalEpochLoss), KLD = np.mean(KLDEpochLoss), pre_loss = np.mean(predictionEpochLoss))
    print("TRAIN:: LOSS: {:.4f}, KLD: {:.4f}, PRE_LOSS: {:.4f}".format(np.mean(totalEpochLoss), np.mean(KLDEpochLoss), np.mean(predictionEpochLoss)))
    return model


def ValidAndTest(taskType, model, device, dataLoader, normalizer):
    model.eval()
    with torch.no_grad():
        predictions = []
        groundTruthes = []
        labelMasks = []
        for data, labelMask in dataLoader:
            data, labelMask = data.cuda(device), labelMask.cuda(device)
            prediction, _, _ = model(data)

            if taskType == "classification":
                prediction = F.sigmoid(prediction)
            if normalizer:
                prediction = normalizer.denorm(prediction)
            predictions.append(prediction.cpu().detach())
            groundTruthes.append(data.y.cpu().detach())
            labelMasks.append(labelMask.cpu().detach())

    predictions = torch.cat(predictions, dim = 0).numpy()
    groundTruthes = torch.cat(groundTruthes, dim = 0).numpy()
    labelMasks = torch.cat(labelMasks, dim = 0).numpy()

    try:
        if taskType == "classification":
            score = LF.ComputeAUCScoreWithMultiClass(predictions, groundTruthes, labelMasks)
        elif taskType == "RMSE":
            score = LF.ComputeRMSEScore(predictions, groundTruthes)
        elif taskType == "MAE":
            score = LF.ComputeMAEScore(predictions, groundTruthes)
    except ValueError as e:
        print("ERROR: A value error occurs when validation or test.\n {}".format(e))
        if taskType == "classification":
            score = 0
        elif taskType in ["MAE", "RMSE"]:
            score = 1e5
    return score

def FineTune(ARGS, taskName, taskType, targetName, taskNum):
    print("INFO: Fine tune stage, {}:{}".format(taskName, targetName))
    MoleculePreModel = VAE(True).cuda(ARGS.device)
    # Load pretrained model
    try:
        if ARGS.runType == "FullStage":
            checkpoint = torch.load("Pretrain{}.pth".format(ARGS.gitNode), weights_only=True)
        else:
            checkpoint = torch.load("Pretrain{}.pth".format(ARGS.checkPointName), weights_only=True)
        MoleculePreModel.load_state_dict(checkpoint)
        print("INFO: Load model parameter success!")
    except(FileNotFoundError, RuntimeError) as e:
        print("ERROR: Load model parameter fail! Exit!")
        print(e)
        exit()
    downstreamModel = DownstreamModel(ARGS, taskNum, MoleculePreModel).cuda(ARGS.device)

    # Prepare dataset
    molTuneDatasetWrapper = MolTuneDatasetWrapper(ARGS, taskName, taskType, targetName)
    trainLoader, valLoader, testLoader = molTuneDatasetWrapper.get_data_loaders()
    print("Train/Valid/Test num: {}/{}/{}".format(len(trainLoader), len(valLoader), len(testLoader)))

    # Set learning rate
    layerList = []
    for name, param in downstreamModel.named_parameters():
        if 'predictionMLP' in name:
            layerList.append(name)
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layerList, downstreamModel.named_parameters()))))
    baseParams = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layerList, downstreamModel.named_parameters()))))
    optimizer = torch.optim.AdamW(
        [{'params': baseParams, 'lr':ARGS.tuningConfig[1]}, {'params': params}],
        ARGS.tuningConfig[0], weight_decay = ARGS.tuningConfig[2])

    
    testScoreList = []
    normalizer = None
    if taskName == "QM7":
        print("Need Normalization")
        labels = []
        for data, _ in trainLoader:
            labels.append(data.y)
        labels = torch.cat(labels)
        normalizer = Normalizer(labels)
        print("Normalizer Mean: {:.4f}/Std: {:.4f}".format(normalizer.mean, normalizer.std))


    for eachInd in range(ARGS.TuningEpoch):
        print("Epoch: {}".format(eachInd))
        downstreamModel = Train(taskType, downstreamModel, ARGS.device, trainLoader, optimizer, normalizer)
        testScore = ValidAndTest(taskType, downstreamModel, ARGS.device, testLoader, normalizer)
        testScoreList.append(testScore)
        print("TEST: Test Score: {:.4f}".format(testScore))
    finalTestScore = -1
    if taskType == "classification":
        epoch = testScoreList.index(max(testScoreList))
        finalTestScore = testScoreList[epoch]
        print(f"{taskName}: Test score: {finalTestScore:.4f}")
    elif taskType in ["MAE", "RMSE"]:
        epoch = testScoreList.index(min(testScoreList))
        finalTestScore = testScoreList[epoch]
        print(f"{taskName}: Test score: {finalTestScore:.4f}")

    del MoleculePreModel
    del downstreamModel
    del optimizer
    return finalTestScore