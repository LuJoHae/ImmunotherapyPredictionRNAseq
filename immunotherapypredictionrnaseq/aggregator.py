import torch.nn as nn
import torch
import torch.nn.functional as F


class GeneSetAggregator(nn.Module):

    def __init__(self, genesets_indices, feature_dim):
        super(GeneSetAggregator, self).__init__()

        self.genesets_indices = genesets_indices
        # Attention weights for each gene in each gene set
        self.attention_weights = nn.ParameterDict(
            {
                geneset_name: nn.Parameter(torch.randn(len(geneset_indices), feature_dim))
                for geneset_name, geneset_indices in self.genesets_indices.items()
            }
        )

    def forward(self, gene_features):
        """
        Forward pass of the module.

        :param gene_features: A tensor of shape (batch_size, num_genes, feature_dim), representing gene-level features.
        :return: A tensor of shape (batch_size, num_gene_sets, feature_dim), representing gene set level features.
        """
        gene_set_features = []
        for geneset_name, geneset_indices in self.genesets_indices.items():
            set_features = gene_features[:, geneset_indices, :]  # Extract features for genes in the set
            attention = F.softmax(
                self.attention_weights[geneset_name].expand(gene_features.size(0), -1, -1),
                dim=1
            )
            weighted_features = set_features * attention
            aggregated_features = torch.sum(weighted_features, dim=1)
            gene_set_features.append(aggregated_features)

        gene_set_features = torch.stack(gene_set_features, dim=1)
        return gene_set_features


class CellPathwayPoolingAggregator(nn.Module):
    def __init__(self, cellpathway_indices):
        """
        Initializes the CellPathwayAggregatorPooling module.
        :param cellpathway_indices: A list of lists, each sublist contains indices of gene sets for a specific cell pathway.
        """
        super(CellPathwayPoolingAggregator, self).__init__()
        self.cellpathway_indices = cellpathway_indices

    def forward(self, gene_set_features):
        """
        Forward pass of the module.
        :param gene_set_features: A tensor of shape (batch_size, num_gene_sets, feature_dim).
        :return: A tensor of shape (batch_size, num_cellpathways).
        """
        aggregated_features = torch.zeros(
            gene_set_features.size(0), len(self.cellpathway_indices), device=gene_set_features.device
        )

        for i, indices in enumerate(self.cellpathway_indices.values()):
            aggregated_features[:, i] = torch.mean(gene_set_features[:, indices], dim=1)

        return aggregated_features
