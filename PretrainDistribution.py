from math import inf
from tqdm import tqdm
from Arguments import argparser
from DataHelper import PretrainDataSetTxtFile
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.utils.data import DistributedSampler
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import numpy as np
import torch
from VAE import VAE
import os
import lossFunctions as LF
import sys
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import torch.nn.functional as F

def PretrainValidation(model, dataLoader, device):
    model.eval()
    model.to(device)

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

def PretrainUpdata(model, dataLoader, device, optimizer):
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
            tqdmIter.set_postfix(RANK = torch.distributed.get_rank(), train_loss=np.mean(trainMolLoss), atom_loss=np.mean(trainAtomLoss), bond_type_loss = np.mean(trainBondTypeLoss), bond_dir_loss = np.mean(trainBondDirLoss),  KLD = np.mean(trainKLD), struLoss = np.mean(trainStruLoss))
    print("TRAIN:: RANK: {}, ATMO_LOSS: {:.4f}, Bond_Type_LOSS: {:.4f}, Bond_Dir_Loss: {:.4f},  KLD: {:.4f}, Stru_Loss:{:.4f}".format(torch.distributed.get_rank(), np.mean(trainAtomLoss), np.mean(trainBondTypeLoss), np.mean(trainBondDirLoss), np.mean(trainKLD), np.mean(trainStruLoss)))
    return model


def SaveCheckpoint(ARGS, model):
    torch.save(model.state_dict(), "Pretrain{}.pth".format(ARGS.gitNode))

def SplitTrainAndValDataset(dataset_len, val_ratio=0.05, seed=42):
    rank = dist.get_rank()
    idx_container = [None, None]                  # [train_idx, val_idx]
    if rank == 0:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(dataset_len, generator=g)
        val_len = int(dataset_len * val_ratio)
        idx_container[1] = perm[:val_len].tolist()
        idx_container[0] = perm[val_len:].tolist()
    # 把 rank0 产生的 idx_container 广播给所有进程
    dist.broadcast_object_list(idx_container, src=0)
    return idx_container  # train_idx, val_idx

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)

def RunDistributionPretrain(ARGS):
    rank        = int(os.environ["RANK"])
    local_rank  = int(os.environ["LOCAL_RANK"])     # 当前进程占用哪块 GPU
    world_size  = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{rank}'

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    dataset = PretrainDataSetTxtFile()
    train_idx, val_idx = SplitTrainAndValDataset(len(dataset), 0.05)
    if rank == 0:
        print("Pretrain train/val:{}/{}".format(len(train_idx), len(val_idx)))
    trainSet = Subset(dataset, train_idx)
    valSet = Subset(dataset, val_idx)

    trainSampler  = DistributedSampler(trainSet, num_replicas=world_size, rank=rank, shuffle=True)
    trainLoader   = DataLoader(trainSet, batch_size=256, sampler=trainSampler, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)

    valLoader = None
    if rank == 0:
        valLoader = DataLoader(valSet, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)

    model = VAE(tuningFlag = False).to(rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr = ARGS.pretrainConfig[0], weight_decay = ARGS.pretrainConfig[1])
    scheduler = CosineAnnealingLR(optimizer, T_max = ARGS.pretrainEpoch - ARGS.pretrainWarmUp, eta_min = 0, last_epoch = -1)

    valAcc = 0

    for eachInd in range(1, ARGS.pretrainEpoch + 1):
        if  rank == 0:
            start_time = time.time()
        try:
            model = PretrainUpdata(model, trainLoader, device, optimizer)
            if eachInd >= ARGS.pretrainWarmUp:
                scheduler.step()
            if rank == 0:
                if eachInd % 2 == 0:
                    molAcc = PretrainValidation(model.module, valLoader, f'cuda:0')
                    print("INFO:: VAL: Epoch: {}, Run Time :{}, Pretrain validation ACC:{:.4f}".format(eachInd, time.time() - start_time, molAcc))
                    if molAcc >= valAcc: # Validation support this epoch is the best
                        print("Save model.")
                        SaveCheckpoint(ARGS, model.module)
                        valAcc = molAcc
                else:
                    print("INFO:: Epoch: {}, Run Time :{}".format(eachInd, time.time() - start_time))
            dist.barrier()
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory error occurs at epoch {}. Saving checkpoint....".format(eachInd))
        torch.cuda.empty_cache() # clean unuse cuda memory

if __name__ == "__main__":
    ARGS = argparser()
    RunDistributionPretrain(ARGS)
    dist.destroy_process_group()


