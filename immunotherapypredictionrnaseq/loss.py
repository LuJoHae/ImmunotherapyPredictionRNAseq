import torch.nn as nn
import torch
import torch.nn.functional as F
from immunotherapypredictionrnaseq.data import ContrastiveTriplet

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x: ContrastiveTriplet) -> torch.Tensor:

        distance_positive = 1 - F.cosine_similarity(x.anchor, x.positive)
        distance_negative = 1 - F.cosine_similarity(x.anchor, x.negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()