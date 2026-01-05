from immunotherapypredictionrnaseq.model import Model, EncoderConfig
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
import torch
from pathlib import Path


def test_model():
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()

    encoder_config = EncoderConfig(input_dim=len(token_config.genes))

    d_model = 6
    torch.manual_seed(0)
    x = torch.rand(d_model, len(token_config.genes) + 1)
    model = Model(encoder_config, token_config)
    geneset_level_proj, cellpathway_level_proj, encoding = model(x)
    assert geneset_level_proj.shape == (d_model, len(token_config.genesets) + 2)
    assert cellpathway_level_proj.shape == (d_model, len(token_config.broad_celltype_pathways) + 2)
    assert encoding.shape == (d_model, len(token_config.genes) + 2, encoder_config.transformer_dim)