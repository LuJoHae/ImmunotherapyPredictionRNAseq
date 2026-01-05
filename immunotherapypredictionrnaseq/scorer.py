import torch.nn as nn


class Scorer(nn.Module):

    def __init__(self, feature_dim):
        """
        Initializes the GeneSetScoreLinear module for gene set score calculation.
        :param feature_dim: The dimension of features for each gene set.
        """
        super(Scorer, self).__init__()
        self.fc = nn.Linear(
            feature_dim, 1
        )

    def forward(self, x):
        """
        Forward pass of the module.
        :param x: A tensor of shape (batch_size, num_gene_sets, feature_dim).
        :return: A tensor of shape (batch_size, num_gene_sets).
        """
        batch_size, num_gene_sets, _ = x.shape
        x = x.view(-1, x.size(-1))  # Flatten the last two dimensions for linear layer
        scores = self.fc(x)
        scores = scores.view(
            batch_size, num_gene_sets
        )  # Reshape to original batch and gene set dimensions
        return scores
