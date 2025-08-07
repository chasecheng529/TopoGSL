
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import lossFunctions as LF
import sys
from rdkit import RDLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from VAE import VAE
from tqdm import tqdm
from DataHelper import PretrainDataSetTxtFile
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
# 禁用 RDKit 的日志
RDLogger.DisableLog('rdApp.*')

def Pretrain(ARGS):
    # build distribute training env
    model = VAE(tuningFlag = False).cuda(ARGS.device)
    optimizer = optim.Adam(model.parameters(), lr = ARGS.pretrainConfig[0], weight_decay = ARGS.pretrainConfig[1])
    scheduler = CosineAnnealingLR(optimizer, T_max = ARGS.pretrainEpoch - ARGS.pretrainWarmUp, eta_min = 0, last_epoch = -1)
    dataset = PretrainDataSetTxtFile()
    train_size = int(len(dataset) * 0.95)
    val_size = len(dataset) - train_size
    print("TRAIN: {}, VAL: {}".format(train_size, val_size))
    train_set, val_set = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(dataset = val_set, batch_size = ARGS.pretrainBatchSize, num_workers = ARGS.numWorks)
    train_loader = DataLoader(dataset = train_set, batch_size = ARGS.pretrainBatchSize, num_workers = ARGS.numWorks, shuffle = True)

    valMolACC = 0
    for eachInd in range(1, ARGS.pretrainEpoch + 1):
        model = PretrainUpdata(model, ARGS.device, train_loader, optimizer)
        if eachInd >= ARGS.pretrainWarmUp:
            scheduler.step()
        molACC = PretrainValidation(model, ARGS.device, val_loader)
        print("INFO:: VAL: Epoch: {}, Pretrain validation ACC:{:.4f}".format(eachInd, molACC))
        if molACC >= valMolACC: # Validation support this epoch is the best
            print("Save model.")
            torch.save(model.state_dict(), "Pretrain{}.pth".format(ARGS.gitNode))
            valMolACC = molACC
        torch.cuda.empty_cache() # clean unuse cuda memory

def PretrainUpdata(model, device, dataLoader, optimizer):
    model.train()

    trainAtomLoss = []
    trainBondTypeLoss = []
    trainBondDirLoss = []
    trainKLD = []
    trainStruLoss = []
    trainMolLoss = []

    structureLossFn = nn.BCEWithLogitsLoss()
    with tqdm(dataLoader, disable = not sys.stdout.isatty()) as tqdmIter:
        for data in tqdmIter:
            data = data.cuda(device)
            prediction, molMu, molVar, reconAtoms, reconEdgeType, reconEdgeDir, preStrucutre = model(data)

            atomTypeLabel = data.x_full[:, 0]
            edgeTypeLabel = data.edge_attr_full[:, 0]
            edgeDirLabel = data.edge_attr_full[:, 1]

            reconAtomLoss = F.cross_entropy(reconAtoms, atomTypeLabel)
            reconEdgeTypeLoss = F.cross_entropy(reconEdgeType, edgeTypeLabel)
            reconEdgeDirLoss = F.cross_entropy(reconEdgeDir, edgeDirLabel)
            molKLDLoss = LF.ComputeKLDLoss(molMu, molVar)
            structurePredictionLoss = structureLossFn(preStrucutre.view(-1), data.y)
            molTotalLoss = structurePredictionLoss + reconAtomLoss + reconEdgeTypeLoss + reconEdgeDirLoss + molKLDLoss * 0.001

            optimizer.zero_grad()
            molTotalLoss.backward()
            optimizer.step()

            trainAtomLoss.append(reconAtomLoss.item())
            trainBondTypeLoss.append(reconEdgeTypeLoss.item())
            trainBondDirLoss.append(reconEdgeDirLoss.item())
            trainKLD.append(molKLDLoss.item())
            trainMolLoss.append(molTotalLoss.item())
            trainStruLoss.append(structurePredictionLoss.item())
            tqdmIter.set_postfix(train_loss=np.mean(trainMolLoss), atom_loss=np.mean(trainAtomLoss), bond_type_loss = np.mean(trainBondTypeLoss), bond_dir_loss = np.mean(trainBondDirLoss),  KLD = np.mean(trainKLD), struLoss = np.mean(trainStruLoss))
    print("TRAIN:: ATMO_LOSS: {:.4f}, Bond_Type_LOSS: {:.4f}, Bond_Dir_Loss: {:.4f},  KLD: {:.4f}, Stru_Loss:{:.4f}".format( np.mean(trainAtomLoss), np.mean(trainBondTypeLoss), np.mean(trainBondDirLoss), np.mean(trainKLD), np.mean(trainStruLoss)))
    return model

def PretrainValidation(model, device, dataLoader):
    model.eval()
    molAccList = []
    struAccList = []
    with torch.no_grad():
        with tqdm(dataLoader, disable = not sys.stdout.isatty()) as tqdmIter:
            for data in tqdmIter:
                data = data.cuda(device)
                _, _, _, _, _, _, preStrucutre = model(data)
                # atomACC = LF.ReconAtomACC(reconAtoms, gtAtom, molMask)
                # adjACC = LF.ReconAdjACC(reconAtomAdj, gtAdj)
                atomACC = 0
                adjACC = 0
                struACC = LF.ReconAdjACC(preStrucutre, data.y)
                molAccList.append((atomACC + adjACC) / 2)
                struAccList.append(struACC)
    print("VAL:: MOL ACC: {:.4f}, STRU ACC: {:.4f}".format(np.mean(molAccList), np.mean(struAccList)))
    # return (np.mean(molAccList) + np.mean(struAccList)) / 2
    return np.mean(struAccList)