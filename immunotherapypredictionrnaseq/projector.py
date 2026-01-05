import torch.nn as nn
import torch
from immunotherapypredictionrnaseq.aggregator import GeneSetAggregator, CellPathwayPoolingAggregator
from immunotherapypredictionrnaseq.scorer import Scorer
from immunotherapypredictionrnaseq.tokenizer import TokenConfig


class DisentangledProjector(nn.Module):

    def __init__(
        self,
        token_config: TokenConfig,
        batch_size: int,
        feature_dim: int
    ):

        super(DisentangledProjector, self).__init__()
        gene_feature_dim = len(token_config.genes)
        self.patient_scorer = Scorer(feature_dim)
        self.cancer_scorer = Scorer(feature_dim)
        self.geneset_scorer = Scorer(feature_dim)
        self.geneset_aggregator = GeneSetAggregator(token_config.geneset_to_gene_indices, feature_dim)
        # self.cellpathway_aggregator = GeneSetAggregator(token_config.celltype_pathway_to_geneset_indicies, feature_dim)
        self.cellpathway_aggregator = CellPathwayPoolingAggregator(token_config.celltype_pathway_to_geneset_indicies)


    def forward(self, x):

        # x size is  B, L+2, 1
        pid_encoding = x[:, 0:1, :]  # take the learnbale patient id token
        cancer_encoding = x[:, 1:2, :]  # take the cancer_type token
        gene_encoding = x[:, 2:, :]  # take the gene encoding

        geneset_feats = self.geneset_aggregator(gene_encoding)
        geneset_scores = self.geneset_scorer(geneset_feats)
        cellpathway_scores = self.cellpathway_aggregator(geneset_scores)

        cancer_scores = self.cancer_scorer(cancer_encoding)
        pid_scores = self.patient_scorer(pid_encoding)

        geneset_proj = torch.cat([pid_scores, cancer_scores, geneset_scores], dim=1)
        cellpathway_proj = torch.cat([pid_scores, cancer_scores, cellpathway_scores], dim=1)

        return geneset_proj, cellpathway_proj