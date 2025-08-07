from sympy import Float, im
from Arguments import argparser
from torch.utils import data
from tqdm import tqdm
import numpy as np
import json
import torch
import pandas as pd
import pyarrow.parquet as pq
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.Chem.rdchem import BondType as BT
import csv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import random
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import networkx as nx
from copy import deepcopy

from dataSpliter import scaffold_split
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.DATIVE]

BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def removeSubgraph(Graph, center, percent=0.25):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed

class PretrainDataSetTxtFile(data.Dataset):
    def __init__(self):
        super(PretrainDataSetTxtFile, self).__init__()
        filePath = "Data/PretrainData/ZINC.txt"
        with open(filePath, 'r') as fo:
            data = [x.strip() for x in fo.readlines()]
        self.data = data
        self.lapPE = AddLaplacianEigenvectorPE(k = 8, attr_name = "pos", is_undirected = True)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index])
        mol = Chem.AddHs(mol)

        molFingerprint = np.array(MACCSkeys.GenMACCSKeys(mol))
        molStructureLables = torch.tensor(molFingerprint, dtype=torch.float)

        numAtoms = mol.GetNumAtoms()
        type_idx = []
        chirality_idx = []
        atomic_number = []
        bonds = mol.GetBonds()

        removeCenter = random.sample(list(range(numAtoms)), 1)[0]

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        # removedGraph, removePart = removeSubgraph(molGraph, removeCenter, 0.25)

        for atom in mol.GetAtoms():
            type_idx.append(atom.GetAtomicNum())
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row_full, col_full, edge_feat_full = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row_full += [start, end]
            col_full += [end, start]
            edge_feat_full.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat_full.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
        edge_index_full = torch.tensor([row_full, col_full], dtype=torch.long)
        edge_attr_full = torch.tensor(np.array(edge_feat_full), dtype=torch.long)

        data_full = Data(x = x, y = molStructureLables, edge_index = edge_index_full, edge_attr = edge_attr_full,  edge_index_full = edge_index_full, edge_attr_full = edge_attr_full, x_full = x)
        try:
            data_full = self.lapPE(data_full)
        except Exception as e:
            print(self.data[index])
            data_full.pos = torch.zeros([x.shape[0], 8])
        
        return data_full

    def __len__(self):
        return len(self.data)
    
class TuningDataSet(data.Dataset):
    def __init__(self, taskName, taskType, targetName):
        super(TuningDataSet, self).__init__()
        self.smiles_data, self.labels, self.labelMask = ReadSMILESFromCSV("Data/DownstreamData/{}.csv".format(taskName), targetName, taskType)
        self.taskType = taskType

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        type_idx = []
        chirality_idx = []
        atomic_number = []

        for atom in mol.GetAtoms():
            type_idx.append(atom.GetAtomicNum())
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long).reshape(-1, 2)

        if self.taskType == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
            labelMask = torch.tensor(self.labelMask[index], dtype = torch.int)
        elif self.taskType == "RMSE" or self.taskType == "MAE":
            y = torch.tensor(self.labels[index] , dtype=torch.float).view(1,-1)
            labelMask = torch.tensor(self.labelMask[index], dtype = torch.int)

        data = Data(x = x, y = y, edge_index = edge_index, edge_attr = edge_attr)

        return data, labelMask

    def __len__(self):
        return len(self.smiles_data)
    
class MolTuneDatasetWrapper(object):
    def __init__(self, ARGS, taskName, taskType, targetName, splitting = 'scaffold', validSize = 0.1, testSize = 0.1, numWorks = 4):
        super(object, self).__init__()
        self.batchSize = ARGS.TuneBatchSize
        self.numWorks = numWorks
        self.validSize = validSize
        self.testSize = testSize
        self.taskName = taskName
        self.taskType = taskType
        self.targetName = targetName
        self.splitting = splitting
        self.args = ARGS
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = TuningDataSet(self.taskName, self.taskType, self.targetName)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.validSize * num_train))
            split2 = int(np.floor(self.testSize * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.validSize, self.testSize)

        trainDataset = Subset(train_dataset, train_idx)
        validDataset = Subset(train_dataset, valid_idx)
        testDataset = Subset(train_dataset, test_idx)

        train_loader = DataLoader(
            trainDataset, batch_size=self.batchSize,
            num_workers = self.numWorks, drop_last=False, shuffle = True
        )
        valid_loader = DataLoader(
            validDataset, batch_size=self.batchSize,
            num_workers = self.numWorks, drop_last=False
        )
        test_loader = DataLoader(
            testDataset, batch_size=self.batchSize,
            num_workers = self.numWorks, drop_last=False
        )

        return train_loader, valid_loader, test_loader

def ReadSMILESFromCSV(dataPath, targetName, taskType):
    smiles_data, labels = [], []
    with open(dataPath) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol != None:
                    smiles_data.append(smiles) # record the SMILES
                    if taskType == 'classification':
                        label = []
                        for eachTarget in targetName:
                            rawLabel = row.get(eachTarget, "")
                            if rawLabel == "":
                                rawLabel = 0.5
                            rawLabel = float(rawLabel)
                            label.append(rawLabel)
                        labels.append(label)
                    elif taskType == "RMSE" or taskType == "MAE":
                        label = []
                        for eachTarget in targetName:
                            rawLabel = row.get(eachTarget, "")
                            rawLabel = float(rawLabel)
                            label.append(rawLabel)
                        labels.append(label)
                    else:
                        ValueError('task must be either regression or classification')
    labels = np.array(labels)
    labelMask = labels != 0.5
    return smiles_data, labels, labelMask





