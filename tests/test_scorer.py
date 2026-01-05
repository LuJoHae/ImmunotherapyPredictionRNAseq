import immunotherapypredictionrnaseq.scorer
import torch


def test_scorer():
    x = torch.rand(2, 3, 4)
    genesetscorer = immunotherapypredictionrnaseq.scorer.Scorer(feature_dim=4)
    y = genesetscorer(x)
    assert y.shape == (2, 3)