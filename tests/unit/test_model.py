from immunotherapypredictionrnaseq.model import Model
from immunotherapypredictionrnaseq.encoder import EncoderConfig
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
import torch
from pathlib import Path


def test_model():
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()

    encoder_config = EncoderConfig(
        input_dim=len(token_config.genes),
        transformer_dim=4,
        transformer_nhead=2,
        transformer_num_layers=1,
        encoder_dropout=0.1
    )

    n_samples = 6
    torch.manual_seed(0)
    x = torch.rand(n_samples, len(token_config.genes) + 1)
    model = Model(encoder_config, token_config)
    geneset_level_proj, cellpathway_level_proj, encoding = model(x)
    assert geneset_level_proj.shape == (n_samples, len(token_config.genesets) + 2)
    assert cellpathway_level_proj.shape == (n_samples, len(token_config.broad_celltype_pathways) + 2)
    assert encoding.shape == (n_samples, len(token_config.genes) + 2, encoder_config.transformer_dim)