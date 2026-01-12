import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from torch.nn.modules.container import ModuleList
from immunotherapypredictionrnaseq.embedder import GeneEmbedding
import copy

def _get_clones(module, N):
    # copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_cancer_types=33,
        input_dim=15672,
        d_model=32,
        num_layers=2,
        nhead=2,
        dropout=0,
        dim_feedforward=64,
    ):
        super().__init__()
        self.num_layers = num_layers
        d_sqrt_inv = 1 / math.sqrt(d_model)
        self.pid_token_embedder = nn.Parameter(torch.randn(1, d_model))
        torch.nn.init.uniform_(self.pid_token_embedder, a=-d_sqrt_inv, b=d_sqrt_inv)
        self.cancer_token_embedder = nn.Embedding(num_cancer_types, d_model)
        self.gene_token_embedder = GeneEmbedding(input_dim, d_model)
        encoder_layer = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=dim_feedforward,
                batch_first=True
            ) for _ in range(num_layers)]
        )
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])


    def forward(self, x):

        cancer_types = x[:, 0].long()  # first column is the cancer type
        genes = x[:, 1:]

        pid_embed = self.pid_token_embedder.expand(x.size(0), -1).unsqueeze(1)  # (B,1, C) CLS Token
        cancer_embed = self.cancer_token_embedder(cancer_types).unsqueeze(1)  # (B,1, C) convert cancer types
        gene_embed = self.gene_token_embedder(genes)  # (B, L, C) gene embeddings

        x = torch.cat([pid_embed, cancer_embed, gene_embed], dim=1)  # B, L+2, C
        for layer in range(self.num_layers):
            x = self.layers[layer](x)

        return x


@dataclass
class EncoderConfig:
    input_dim: int
    encoder_dropout: float
    transformer_dim: int
    transformer_num_layers: int
    transformer_nhead: int
