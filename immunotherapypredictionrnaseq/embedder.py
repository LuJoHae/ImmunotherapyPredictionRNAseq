import torch.nn as nn
from torch import Tensor
import math
from immunotherapypredictionrnaseq.utils import check_params_and_gradients


class GeneEmbedding(nn.Module):
    def __init__(self, n_genes: int = 876, embed_dim: int = 32) -> None:
        super().__init__()
        d_sqrt_inv = 1 / math.sqrt(embed_dim)
        self.weight = nn.Parameter(Tensor(n_genes, embed_dim))
        nn.init.uniform_(self.weight, a=-d_sqrt_inv, b=d_sqrt_inv)
        self.bias = nn.Parameter(Tensor(n_genes, embed_dim))
        nn.init.uniform_(self.bias, a=-d_sqrt_inv, b=d_sqrt_inv)
        self.activation = nn.ReLU()
        check_params_and_gradients(self)

    @property
    def n_genes(self) -> int:
        """The number of genes."""
        return len(self.weight)

    @property
    def d_gene(self) -> int:
        """The size of one gene."""
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
          x: embeddings (batch_size, seq_length)

        Returns:
            gene value embeddings (batch_size, seq_length, d_model) + gene positional embedding (batch_size, seq_length, d_model)
        """
        x = self.weight[None] * x[..., None]
        x = x + self.bias[None]
        x = self.activation(x)
        return x
