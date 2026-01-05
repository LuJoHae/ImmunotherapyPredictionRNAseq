import torch.nn as nn
from dataclasses import dataclass

from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.utils import fixseed
from immunotherapypredictionrnaseq.encoder import TransformerEncoder
from immunotherapypredictionrnaseq.projector import DisentangledProjector


@dataclass
class EncoderConfig:
    input_dim: int
    encoder_dropout: float = 0.0
    transformer_dim: int = 32
    transformer_num_layers: int = 1
    transformer_nhead: int = 2


class Model(nn.Module):

    def __init__(
        self,
        encoder_config: EncoderConfig,
        token_config: TokenConfig,
        seed: int = 0
    ):
        super().__init__()
        fixseed(seed)

        self.inputencoder = TransformerEncoder(
            num_cancer_types=len(token_config.cancer_types),
            input_dim=encoder_config.input_dim,
            d_model=encoder_config.transformer_dim,
            num_layers=encoder_config.transformer_num_layers,
            nhead=encoder_config.transformer_nhead,
            dropout=encoder_config.encoder_dropout
        )

        self.projector = DisentangledProjector(
            feature_dim=encoder_config.transformer_dim,
            batch_size=encoder_config.input_dim,
            token_config=token_config
        )

    def forward(self, x):

        # output： B,L+2, (dataset:1, cancer:1, gene),C
        encoding = self.inputencoder(x)
        geneset_level_proj, cellpathway_level_proj = self.projector(encoding)
        return (geneset_level_proj, cellpathway_level_proj, encoding)
