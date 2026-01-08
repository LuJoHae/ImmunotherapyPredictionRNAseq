import random

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

from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from icir.datasets import ImmuneCheckpointTherapyResponseProcessedGeneNormalizedClinicalDataNormalized as ICTR


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
        self._tcga_data = None
        self._icir_data = None
        self.genes = None
        self.var = None
        self.device = torch.device("cpu")
        self._scaler = StandardScaler()

    def __len__(self):
        return len(self._tcga_data)

    def load(self, n: int = 0, cache: Optional[Path] = None, alpha=1e7):
        if cache and cache.exists() and cache.is_dir() and any(cache.iterdir()):
            tcga_data = self._load_from_cache(cache.joinpath("tcga_data.npy"), n)
            icir_data = self._load_from_cache(cache.joinpath("icir_data.npy"), 0)
            response_data = self._load_from_cache(cache.joinpath("response_data.npy"), 0)
        else:
            tcga_adata = self._load_tcga_data_from_lair(n)
            assert tcga_adata.n_obs > 0
            icir_adata = self._load_icir_data_from_lair()
            try:
                x = np.concatenate([tcga_adata.X.copy(), icir_adata.X.copy()], axis=0)
            except Exception as e:
                print(tcga_adata.X.copy().shape, icir_adata.X.copy().shape)
                raise e

            col_medians = np.nanmedian(x, axis=0)
            inds = np.where(np.isnan(x))
            x[inds] = np.take(col_medians, inds[1])

            x = Normalizer(norm="l1").fit_transform(x)
            x = np.log1p(alpha * x)
            x = StandardScaler().fit_transform(x)
            tcga_adata = ad.AnnData(X=x[:tcga_adata.n_obs, :], obs=tcga_adata.obs, var=tcga_adata.var)
            icir_adata = ad.AnnData(X=x[tcga_adata.n_obs:, :], obs=icir_adata.obs, var=icir_adata.var)
            response_data = np.array(icir_adata.obs["response"].map(lambda response: 1 if response == "R" else 0)).astype(np.int64)
            tcga_cancer_codes = self._get_cancer_codes(tcga_adata)
            tcga_data = np.concatenate([tcga_cancer_codes, tcga_adata.X], axis=1)
            icir_cancer_codes = self._get_cancer_codes(icir_adata)
            icir_data = np.concatenate([icir_cancer_codes, icir_adata.X], axis=1)
            if cache:
                self._write_to_cache(cache.joinpath("tcga_data.npy"), tcga_data)
                self._write_to_cache(cache.joinpath("icir_data.npy"), icir_data)
                self._write_to_cache(cache.joinpath("response_data.npy"), response_data)
        if n != 0:
            assert tcga_data.shape[0] == n, f"Expected {n} samples, got {tcga_data.shape[0]}"
        self._tcga_data = torch.tensor(tcga_data, dtype=torch.float32).clone().detach()
        self._icir_data = torch.tensor(icir_data, dtype=torch.float32).clone().detach()
        self._response_data = torch.tensor(response_data, dtype=torch.float32).clone().detach()
        self._is_loaded = True

    def _write_to_cache(self, cache_file: Path, data: np.ndarray) -> None:
        np.save(cache_file, data)

    def _load_from_cache(self, cache_file: Path, n: int) -> np.ndarray:
        if n == 0:
            return np.load(cache_file)
        return np.load(cache_file)[:n]

    def __getitem__(self, idx: int | Iterable[int]) -> ContrastiveTriplet:
        if not self._is_loaded:
            raise RuntimeError("Call load() before using this method.")
        idx_len = (
            len(range(*idx.indices(self._tcga_data.shape[0]))) if isinstance(idx, slice) else
            len(idx) if isinstance(idx, Iterable) else
            None
        )
        random_idx = np.random.randint(0, len(self._tcga_data), size=idx_len)
        x = ContrastiveTriplet(
            anchor=self._tcga_data[idx],
            positive=self.augment(self._tcga_data[idx]),
            negative=self.augment(self._tcga_data[random_idx])
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
        self._tcga_data = self._tcga_data.to(device)
        self.device = device

    def _get_cancer_codes(self, adata_combined):
        cancer_codes = np.array([self._token_config.cancer_type_to_code[cancer_type] for cancer_type in
                                 adata_combined.obs["cancer_type"]]).reshape(-1, 1)
        assert cancer_codes.dtype == np.int64, cancer_codes.dtype
        return cancer_codes

    def _load_tcga_data_from_lair(self, n):
        adata_combined = self._load_tcga_adata_from_lair(n)
        assert adata_combined.n_obs > 0
        adata_combined = self._process_tcga_adata_metadata(adata_combined)
        assert adata_combined.n_obs > 0
        return adata_combined

    def _process_tcga_adata_metadata(self, adata_combined):
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

    def _load_tcga_adata_from_lair(self, n):
        adatas = []
        for filename, path in self._lair.get_dataset_filepaths(tcga.AllProjectsAdata()).items():
            adata = ad.read_h5ad(path, backed="r")
            adata = adata[:, :] if n == 0 else adata[:, :n]  # random.sample(range(adata.n_vars), n)]
            adata = adata.to_memory().T
            adata.X = adata.X.astype(np.int64)
            adata.obs["cancer_type"] = filename.split(".")[0]
            adatas.append(adata)
        adata_combined = ad.concat(adatas)
        if n != 0:
            adata_combined = adata_combined[random.sample(list(adata_combined.obs_names), k=n), :].copy()
        return adata_combined.copy()

    def _load_icir_data_from_lair(self):
        adata_combined = self._load_icir_adata_from_lair()
        adata_combined = self._process_icir_adata_metadata(adata_combined)
        return adata_combined

    def _load_icir_adata_from_lair(self):
        adatas = []
        cancer_types = {
            "Auslander.h5ad": "SKCM",
            "Chen-CTLA4.h5ad": "SKCM",
            "Chen-PD1.h5ad": "SKCM",
            "Freeman.h5ad": "SKCM",
            "Gide.h5ad": "SKCM",
            "Hugo.h5ad": "SKCM",
            "Lauss.h5ad": "SKCM",
            "Liu.h5ad": "SKCM",
            "Prat.h5ad": "SKCM",  # also contains other cancers
            "Ravi.h5ad": "KIRC",
            "Riaz.h5ad": "SKCM",
            "Rose.h5ad": "BLCA",
            "Snyder.h5ad": "BLCA",
            "VanAllen.h5ad": "SKCM"
        }
        for filename, path in self._lair.get_dataset_filepaths(ICTR()).items():
            adata = ad.read_h5ad(path, backed="r")
            adata = adata.to_memory()
            adata.X = adata.X.astype(np.int64)
            adata.obs["cancer_type"] = cancer_types[path.name]
            adatas.append(adata)
        adata_combined = ad.concat(adatas, join="outer")
        return adata_combined

    def _process_icir_adata_metadata(self, adata_combined):
        ensembl = pyensembl.EnsemblRelease(111, "human")
        adata_combined.var["gene_name"] = [ensembl.gene_name_of_gene_id(id) for id in adata_combined.var_names]
        adata_combined = adata_combined[:, list(adata_combined.var["gene_name"].isin(self._token_config.genes))]
        assert adata_combined.shape == (937, 912)
        assert adata_combined.var["gene_name"].nunique() == 912
        adata_combined.var.set_index("gene_name", inplace=True)
        adata_combined = adata_combined[:, sorted(list(adata_combined.var_names))]
        assert adata_combined.shape == (937, 912)
        return adata_combined
