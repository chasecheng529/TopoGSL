import torch.nn as nn
import torch
import math

class DownstreamModel(nn.Module):
    def __init__(self, ARGS, taskNum, moleculeEncoder):
        super(DownstreamModel, self).__init__()
        self.moleculeEncoder = moleculeEncoder
        taskDim = taskNum
        self.predictionMLP = nn.Sequential(
            nn.Linear(ARGS.FeatureNums // 2, ARGS.FeatureNums // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ARGS.FeatureNums // 4, taskDim)
        )
    
    def forward(self, data):
        # compute latent features from molecule graph and bond angle graph
        molLatentFeature, molMu, molVar = self.moleculeEncoder(data)

        prediction = self.predictionMLP(molLatentFeature)
        return prediction, molMu, molVar