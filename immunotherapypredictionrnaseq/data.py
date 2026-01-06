import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Self
import pyensembl
import datalair
import tcga
from typing import Optional
from sklearn.preprocessing import Normalizer, StandardScaler
from scipy.stats import skew, kurtosis

from immunotherapypredictionrnaseq.tokenizer import TokenConfig


def collate_triplets(batch):
    # batch is a list of ContrastiveTriplet objects
    anchors = torch.stack([b.anchor for b in batch])
    positives = torch.stack([b.positive for b in batch])
    negatives = torch.stack([b.negative for b in batch])
    return ContrastiveTriplet(anchors, positives, negatives)


@dataclass(frozen=True)
class ContrastiveTriplet:
    anchor: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor

    def __post_init__(self):
        if not (self.anchor.shape == self.positive.shape == self.negative.shape):
            raise ValueError(
                "All tensors must have the same shape: {}, {}, {}"
                .format(self.anchor.shape, self.positive.shape, self.negative.shape)
            )
        object.__setattr__(self, 'shape', self.anchor.shape)

    def to_dict(self):
        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative
        }

    def __repr__(self):
        return f"ContrastiveTriplet(\n\tanchor   = {self.anchor},\n\tpositive = {self.positive},\n\tnegative = {self.negative}\n)"

    def apply(self, func: callable) -> Self:
        return ContrastiveTriplet(
            anchor=func(self.anchor),
            positive=func(self.positive),
            negative=func(self.negative)
        )


def gene_from_id(gene_id, ensembl):
    try:
        gene_name = ensembl.gene_by_id(gene_id.split('.')[0]).gene_name
    except ValueError as e:
        if "GTF database needs to be created" in str(e):
            raise e
        return None
    if gene_name in (None, ""):
        return None
    return gene_name


class TCGAData(Dataset):
    def __init__(
            self,
            lair_path: str | Path,
            token_config: TokenConfig,
            additive_noise_level: float = 1,
            multiplicative_noise_level: float = 0.10,
            drop_rate: float = 0.05
    ):
        lair_path = Path(lair_path)
        assert lair_path.exists()
        lair = datalair.Lair(lair_path)
        lair.assert_ok_satus()
        self._lair = lair
        self._token_config = token_config
        self._is_loaded = False
        self._additive_noise_level = additive_noise_level
        self._multiplicative_noise_level = multiplicative_noise_level
        self._drop_rate = drop_rate
        self._data = None
        self.genes = None
        self.var = None
        self.device = torch.device("cpu")
        self._scaler = StandardScaler()


    def __len__(self):
        return len(self._data)


    def load(self, n: int = 0, cache: Optional[Path] = None, alpha=1e7):
        if cache and cache.exists():
            data = self._load_from_cache(cache, n)
        else:
            adata = self._load_from_lair(n)
            x = Normalizer(norm="l1").fit_transform(adata.X.copy())
            x = np.log1p(alpha * x)
            x = StandardScaler().fit_transform(x)
            adata = ad.AnnData(X=x, obs=adata.obs, var=adata.var)
            cancer_codes = self._get_cancer_codes(adata)
            data = np.concatenate([cancer_codes, adata.X], axis=1)
            if cache:
                self._write_to_cache(cache, data)
        self._assert_data_is_gaussian(data, n)
        self._data = torch.tensor(data, dtype=torch.float32).clone().detach()
        self._is_loaded = True


    def _assert_data_is_gaussian(self, data, n):
        if n == 0:
            assert np.isclose(data[:, 1:].flatten().mean(), 0)
            assert np.isclose(data[:, 1:].flatten().var(), 1)
            assert np.abs(skew(data[:, 1:].flatten())) <= 0.1
            assert np.abs(kurtosis(data[:, 1:].flatten())) <= 1.5
        else:
            assert np.abs(data[:, 1:].flatten().mean()) <= 0.1
            assert data[:, 1:].flatten().var() - 1 <= 0.1


    def _write_to_cache(self, cache_file: Path, data: np.ndarray) -> None:
        np.save(cache_file, data)


    def _load_from_cache(self, cache_file: Path, n: int) -> np.ndarray:
        return np.load(cache_file)[:n]


    def _load_from_lair(self, n):
        adata_combined = self._load_adata_from_lair(n)
        adata_combined = self._process_adata_metadata(adata_combined)
        return adata_combined

    def _get_cancer_codes(self, adata_combined):
        cancer_codes = np.array([self._token_config.cancer_type_to_code[cancer_type] for cancer_type in
                                 adata_combined.obs["cancer_type"]]).reshape(-1, 1)
        assert cancer_codes.dtype == np.int64, cancer_codes.dtype
        return cancer_codes

    def _process_adata_metadata(self, adata_combined):
        self.genes = adata_combined.var_names.tolist()
        ensembl = pyensembl.EnsemblRelease(111)
        gene_names = [(gene_from_id(gene_id.split('.')[0], ensembl)) for gene_id in self.genes]
        adata_combined.var["gene_name"] = gene_names
        adata_combined = adata_combined[:, ~adata_combined.var["gene_name"].isna()]
        adata_combined.var.set_index("gene_name", inplace=True)
        adata_combined = adata_combined.copy()
        adata_combined.var_names_make_unique()
        adata_combined = adata_combined[:, self._token_config.genes.tolist()]
        adata_combined.X = adata_combined.X.astype(np.int64)
        assert isinstance(adata_combined.X, np.ndarray)
        assert adata_combined.X.dtype == np.int64, adata_combined.X.dtype
        return adata_combined

    def _load_adata_from_lair(self, n):
        adatas = []
        for filename, path in self._lair.get_dataset_filepaths(tcga.AllProjectsAdata()).items():
            adata = ad.read_h5ad(path, backed="r")
            adata = adata[:, :] if n == 0 else adata[:, :n]  # random.sample(range(adata.n_vars), n)]
            adata = adata.to_memory().T
            adata.X = adata.X.astype(np.int64)
            adata.obs["cancer_type"] = filename.split(".")[0]
            adatas.append(adata)
        adata_combined = ad.concat(adatas)
        return adata_combined

    def __getitem__(self, idx: int | Iterable[int]) -> ContrastiveTriplet:
        if not self._is_loaded:
            raise RuntimeError("Call load() before using this method.")
        idx_len = (
            len(range(*idx.indices(self._data.shape[0]))) if isinstance(idx, slice) else
            len(idx) if isinstance(idx, Iterable) else
            None
        )
        random_idx = np.random.randint(0, len(self._data), size=idx_len)
        x = ContrastiveTriplet(
            anchor=self._data[idx],
            positive=self.augment(self._data[idx]),
            negative=self.augment(self._data[random_idx])
        )
        return x


    def augment(self, x: torch.Tensor) -> torch.Tensor:
        assert x.device == self.device, "x.device={}; self.device={}".format(x.device, self.device)
        add_noise = torch.distributions.Normal(loc=0, scale=self._additive_noise_level).sample(x.shape).to(self.device) \
            if self._additive_noise_level > 0 else torch.tensor(0, device=self.device)
        mult_noise = torch.distributions.Normal(loc=1, scale=self._multiplicative_noise_level).sample(x.shape).to(self.device) \
            if self._multiplicative_noise_level > 0 else torch.tensor(1, device=self.device)
        dropout = torch.distributions.Bernoulli(probs=1 - self._drop_rate).sample(x.shape).to(self.device) \
            if self._drop_rate > 0 else torch.tensor(1, device=self.device)

        cancer_idx = (slice(None), 0) if x.ndim >= 2 else (0,)
        add_noise[cancer_idx], mult_noise[cancer_idx], dropout[cancer_idx] = 0, 1, 1
        x = torch.clamp(dropout * (mult_noise * x + add_noise), min=0.0)
        return x


    def to(self, device: torch.device) -> None:
        self._data = self._data.to(device)
        self.device = device
