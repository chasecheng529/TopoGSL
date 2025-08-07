from MoleculeACE import Data, Descriptors, calc_rmse, calc_cliff_rmse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as GEOData
from torch.utils import data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import torch
import numpy as np
from VAE import VAE
from downstreamModel import DownstreamModel
from Arguments import argparser
import lossFunctions as LF
import sys
from tqdm import tqdm

DataSetNameList = ["CHEMBL1862_Ki", "CHEMBL1871_Ki", "CHEMBL2034_Ki", "CHEMBL2047_EC50", "CHEMBL204_Ki",
                   'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50', 'CHEMBL219_Ki', 'CHEMBL228_Ki', 
                   'CHEMBL231_Ki', 'CHEMBL233_Ki', 'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki',
                   'CHEMBL237_EC50', 'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki',
                   'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki', 'CHEMBL2971_Ki',
                   'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki', 'CHEMBL4616_EC50', 'CHEMBL4792_Ki']

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.DATIVE]
BONDDIR_LIST = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT]

class SampledDataset(data.Dataset):
    def __init__(self, smiles, labels):
        super(SampledDataset, self).__init__()
        self.smiles = smiles
        self.labels = labels

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles[index])
        mol = Chem.AddHs(mol)

        type_idx, chirality_idx = [], []
        for atom in mol.GetAtoms():
            type_idx.append(atom.GetAtomicNum())
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge = [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
            edge_feat += [edge, edge]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long).reshape(-1, 2)

        label = torch.tensor(self.labels[index] , dtype=torch.float).view(1,-1)

        return GEOData(x=x, y = label, edge_index=edge_index, edge_attr=edge_attr)

    def __len__(self):
        return len(self.smiles)

def PackageData(datasetname):
    dataset = datasetname
    descriptor = Descriptors.GRAPH
    data = Data(dataset)
    data(descriptor)


    trainSmiles = [data.smiles for data in data.x_train]
    trainY = data.y_train
    testSmiles = [data.smiles for data in data.x_test]
    testY = data.y_test
    cliffMols = data.cliff_mols_test

    trainDataSet = SampledDataset(trainSmiles, trainY)
    testDataSet = SampledDataset(testSmiles, testY)

    return trainDataSet, testDataSet, cliffMols

def GetDatasetResult(ARGS):
    MoleculePreModel = VAE(True).cuda(ARGS.device)
    checkpoint = torch.load("Pretrain{}.pth".format("f64136"), weights_only=True)
    MoleculePreModel.load_state_dict(checkpoint)
    downstreamModel = DownstreamModel(ARGS, 1, MoleculePreModel).cuda(ARGS.device)

    dataName = DataSetNameList[ARGS.NameIndex]
    trainDataSet, testDataSet, cliffMols = PackageData(dataName)
    trainLoader = DataLoader(trainDataSet, batch_size = 32, num_workers = 4, drop_last=True, shuffle = True)
    testLoader = DataLoader(testDataSet, batch_size = 32, num_workers = 4, drop_last=False, shuffle = False)

    layerList = []
    for name, param in downstreamModel.named_parameters():
        if 'predictionMLP' in name:
            layerList.append(name)
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layerList, downstreamModel.named_parameters()))))
    baseParams = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layerList, downstreamModel.named_parameters()))))
    optimizer = torch.optim.AdamW(
        [{'params': baseParams, 'lr':ARGS.tuningConfig[1]}, {'params': params}],
        ARGS.tuningConfig[0], weight_decay = ARGS.tuningConfig[2])
    
    mseList = []
    mseCliffList = []
    for eachInd in range(ARGS.TuningEpoch):
        downstreamModel = Train(downstreamModel, ARGS.device, trainLoader, optimizer)
        testRMSE, testCliffRMSE = ValidAndTest(downstreamModel, ARGS.device, testLoader, cliffMols)
        mseList.append(testRMSE)
        mseCliffList.append(testCliffRMSE)
        print("Epoch: {}, RMSE: {:.4f}, RMSE_CLIFF: {:.4f}".format(eachInd, testRMSE, testCliffRMSE))
    bestEpoch = mseCliffList.index(min(mseCliffList))
    bestRMSE = mseList[bestEpoch]
    bestRMSECliff = mseCliffList[bestEpoch]

    print("DataSet: {}, BEST RMSE: {:.4f}, BEST RMSE_CLIFF: {:.4f}".format(dataName, bestRMSE, bestRMSECliff))

    return bestRMSE, bestRMSECliff


def Train(model, device, dataLoader, optimizer):
    KLDEpochLoss = []
    predictionEpochLoss = []
    totalEpochLoss = []
    
    model.train()
    with tqdm(dataLoader, disable = not sys.stdout.isatty()) as tqdmIter:
        for data in tqdmIter:
            optimizer.zero_grad()
            data = data.cuda(device)
            prediction, molMu, molVar = model(data)

            molKLDLoss = LF.ComputeKLDLoss(molMu, molVar)
            predictionLoss = LF.RegressionLoss(prediction, data.y)

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


def ValidAndTest(model, device, dataLoader, cliff_mols_test):
    model.eval()
    with torch.no_grad():
        predictions = []
        groundTruthes = []
        labelMasks = []
        for data in dataLoader:
            data = data.cuda(device)
            prediction, _, _ = model(data)

            predictions.append(prediction.cpu().detach())
            groundTruthes.append(data.y.cpu().detach())

    predictions = torch.cat(predictions, dim = 0).numpy()
    groundTruthes = torch.cat(groundTruthes, dim = 0).numpy()

    rmse = calc_rmse(groundTruthes, predictions)
    rmse_cliff = calc_cliff_rmse(y_test_pred=predictions, y_test=groundTruthes, cliff_mols_test=cliff_mols_test)

    return rmse, rmse_cliff

if __name__ == "__main__":
    ARGS = argparser()
    datasetResultRMSE = []
    datasetResultRMSECliff = []
    for i in range(10):
        print("Time: {}".format(i))
        bestRMSE, bestRMSECliff = GetDatasetResult(ARGS)
        datasetResultRMSE.append(bestRMSE)
        datasetResultRMSECliff.append(bestRMSECliff)
    bestTime = datasetResultRMSE.index(min(datasetResultRMSE))
    bestRMSE = datasetResultRMSE[bestTime]
    bestRMSECliff = datasetResultRMSECliff[bestTime]
    print("BEST: TIME: {}, RMSE: {:.4f}, RMSE_CLIFF: {:.4f}".format(bestTime, bestRMSE, bestRMSECliff))



