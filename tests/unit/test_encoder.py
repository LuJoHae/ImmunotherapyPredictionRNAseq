import torch

import immunotherapypredictionrnaseq.encoder as encoder

def test_transformer_encoder():
    model = encoder.TransformerEncoder(num_layers=2, num_cancer_types=4, input_dim=10, d_model=8, nhead=2, dropout=0.1, dim_feedforward=8)
    cancer_labels = torch.randint(low=0, high=4, size=(100,1))
    x = torch.randn(size=(100, 10))
    x = torch.cat([cancer_labels, x], dim=1)
    assert x.shape == (100, 11)
    model(x)
