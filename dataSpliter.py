from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
import sys
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem

def genScaffold(smiles, includeChirality = True):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles = smiles, includeChirality = includeChirality)
    return scaffold


def RandomSplit(dataset, seed = 45, trainFrac = 0.8, validFrac = 0.1, testFrac = 0.1):
    datasetSize = len(dataset)
    indexes = list(range(datasetSize))
    randomRange = np.random.RandomState(seed)
    randomRange.shuffle(indexes)
    trainCutoff = int(trainFrac * datasetSize)
    validTrainCutoff = int((trainFrac + validFrac) * datasetSize)

    trainDataset = Subset(dataset, indexes[:trainCutoff])
    validDataset = Subset(dataset, indexes[trainCutoff:validTrainCutoff])
    testDataset = Subset(dataset, indexes[validTrainCutoff:])

    return trainDataset, validDataset, testDataset


def ScaffoldSplitter(dataset, trainFrac = 0.8, validFrac = 0.1, testFrac = 0.1):
    datasetSize = len(dataset)

    # create scaffold dict as {"scaffold":[1,2,3...],...}
    scaffoldsDict = {}
    for i in tqdm(range(datasetSize), desc = "Scaffold Spliting", disable = not sys.stdout.isatty()):
        scaffold = genScaffold(dataset[i][0]) # the first return value is smiles
        if scaffold is not None:
            if scaffold not in scaffoldsDict:
                scaffoldsDict[scaffold] = [i]
            else:
                scaffoldsDict[scaffold].append(i)
        else:
            print("ERROR:: Invalid SMILES!")

    # sort the dict from largest to smallest sets
    scaffoldsDict = {key: sorted(value) for key, value in scaffoldsDict.items()}
    allScaffoldsSet = [scaffoldSet for (scaffold, scaffoldSet) in sorted(scaffoldsDict.items(), key = lambda x: (len(x[1]), x[1][0]), reverse = True)]

    # get train, valid, test data index
    trainSize = trainFrac * datasetSize
    validTrainSize = (trainFrac + validFrac) * datasetSize
    trainIndex, validIndex, testIndex = [], [], []
    for scaffoldSet in allScaffoldsSet:
        if len(trainIndex) + len(scaffoldSet) > trainSize: # this scaffold set can not put in train set
            if len(trainIndex) + len(validIndex) + len(scaffoldSet) > validTrainSize: # this scaffold set can not put in valid set
                testIndex += scaffoldSet
            else:
                validIndex += scaffoldSet
        else:
            trainIndex += scaffoldSet

    assert len(set(trainIndex).intersection(set(validIndex))) == 0
    assert len(set(testIndex).intersection(set(validIndex))) == 0

    # get train, valid, test dataset
    trainDataset = Subset(dataset, trainIndex)
    validDataset = Subset(dataset, validIndex)
    testDataset = Subset(dataset, testIndex)

    return trainDataset, validDataset, testDataset

def generate_scaffold(smiles, include_chirality=True):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffolds(dataset):
    scaffolds = {}
    data_len = len(dataset)
    print("Dataset Size: {}".format(data_len))
    with tqdm(dataset.smiles_data,  desc = "Scaffold Spliting", disable = not sys.stdout.isatty()) as smilesLoader:
        for ind, smiles in enumerate(smilesLoader):
            scaffold = generate_scaffold(smiles)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds