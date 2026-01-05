from immunotherapypredictionrnaseq.aggregator import GeneSetAggregator
from pathlib import Path
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
import torch


def test_geneset_aggregator():
    config_path = Path("~").expanduser().joinpath("ImmunotherapyPredictionRNAseq").joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    config = TokenConfig(config_path)
    config.load_config()

    batch_size, feature_dim = 10, 64
    gsa = GeneSetAggregator(config.geneset_to_gene_indices, feature_dim)
    x = torch.rand(batch_size, len(config.genes), feature_dim)
    assert gsa(x).shape == torch.Size([batch_size, len(config.genesets), feature_dim])
