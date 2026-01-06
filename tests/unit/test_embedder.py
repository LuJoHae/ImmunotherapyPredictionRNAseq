import immunotherapypredictionrnaseq.embedder
import torch

def test_embedder():
    embedder = immunotherapypredictionrnaseq.embedder.GeneEmbedding(n_genes = 7, embed_dim = 4)
    x = torch.rand(2, 7)            # (batch_size, number_of_genes)
    y = embedder(x)
    assert y.shape == (2, 7, 4)     # (batch_size, number_of_genes, embedding_dim)