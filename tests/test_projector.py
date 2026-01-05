from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.projector import DisentangledProjector
from pathlib import Path
import torch


def test_disentangled_projector():
    config_path = Path("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()
    B, L, d = 10, len(token_config.genes), 64
    x = torch.rand(B, L+2, d) # add 2 for cancer token and patient token (CLS tokens like in BERT)
    projector = DisentangledProjector(token_config=token_config, batch_size=B, feature_dim=d)
    geneset_proj, cellpathway_proj = projector(x)
    assert geneset_proj.shape == (B, len(token_config.genesets)+2)
    assert cellpathway_proj.shape == (B, len(token_config.broad_celltype_pathways)+2)