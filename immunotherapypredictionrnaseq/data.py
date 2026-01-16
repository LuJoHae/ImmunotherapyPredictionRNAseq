import logging
import random
import copy
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
from enum import Enum

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


class TCGADataStatus(Enum):
    SUPERVISED = "supervised"
    SELFSUPERVISED = "selfsupervised"


class TCGAData(Dataset):
    def __init__(
            self,
            lair_path: str | Path,
            token_config: TokenConfig,
            noise_anchor: float = 0.0,
            noise_positive: float = 0.05,
            noise_negative: float = 0.05,
            mode: str = "pretrain",
    ):
        lair_path = Path(lair_path)
        assert lair_path.exists()
        lair = datalair.Lair(lair_path)
        lair.assert_ok_satus()
        self._lair = lair
        self._token_config = token_config
        self._is_loaded = False
        self.noise_anchor = noise_anchor
        self.noise_positive = noise_positive
        self.noise_negative = noise_negative
        self._data = None
        self._full_data = None
        self.genes = None
        self.var = None
        self.device = torch.device("cpu")
        self._scaler = StandardScaler()
        self._status = TCGADataStatus.SUPERVISED if mode == "finetune" else TCGADataStatus.SELFSUPERVISED
        self.only_gide = False


    def __len__(self):
        return len(self._data)


    def load(self, n: int = 0, cache: Optional[Path] = None, alpha=1e7):
        if cache and cache.exists() and cache.is_dir() and any(cache.iterdir()):
            from_back = True if self._status == TCGADataStatus.SUPERVISED else False
            x = self._load_from_cache(cache.joinpath("x.npy"), n, from_back=from_back)
        else:
            tcga_adata = self._load_tcga_data_from_lair(n=0)
            assert tcga_adata.n_obs > 0
            icir_adata = self._load_icir_data_from_lair(n=0)
            x = np.concatenate([tcga_adata.X.copy(), icir_adata.X.copy()], axis=0, dtype=np.float32)

            # fill NaNs (dataleakage; fix later!)
            col_medians = np.nanmedian(x, axis=0)
            inds = np.where(np.isnan(x))
            x[inds] = np.take(col_medians, inds[1])

            x = Normalizer(norm="l1").fit_transform(x)
            x = np.log1p(alpha * x)
            x = StandardScaler().fit_transform(x)

            response_data = np.concatenate([
                np.repeat(np.nan, tcga_adata.n_obs),
                np.array(icir_adata.obs["response"].map(
                    lambda response: 1 if response == "R" else 0)
                ).astype(np.float32)
            ]).reshape(-1, 1)
            cancer_codes = np.concatenate([
                self._get_cancer_codes(tcga_adata), self._get_cancer_codes(icir_adata)
            ], dtype=np.float32)
            x = np.concatenate([response_data, cancer_codes, x], axis=1, dtype=np.float32)
            if cache:
                self._write_to_cache(cache.joinpath("x.npy"), x)
            if n != 0:
                x = x[np.random.choice(x.shape[0], n, replace=False), :]
        assert x.shape[0] > 0, x.shape
        self._full_data = torch.tensor(x, dtype=torch.float32).clone().detach()
        assert self._full_data.shape[0] > 0
        self.set_data()
        self._is_loaded = True

    def set_status(self, status: TCGADataStatus) -> None:
        self._status = status

    def set_data(self):
        assert self._full_data.shape[0] > 0, self._full_data.shape[0]
        match self._status:
            case TCGADataStatus.SELFSUPERVISED:
                self._data = self._full_data[:, 1:]
            case TCGADataStatus.SUPERVISED:
                logging.info("Loading supervised data. NaNs will be removed removed. {}".format(self._full_data.shape))
                assert self._full_data.shape[0] > 0
                self._data = self._full_data[~np.isnan(self._full_data[:, 0].numpy()), :]
                assert self._data.shape[0] > 0, "No samples found in supervised data."
        self.to(self.device)

    def get_status(self) -> TCGADataStatus:
        return self._status

    def _write_to_cache(self, cache_file: Path, data: np.ndarray) -> None:
        np.save(cache_file, data)

    def _load_from_cache(self, cache_file: Path, n: int, from_back: bool) -> np.ndarray:
        if n == 0:
            return np.load(cache_file)
        if from_back:
            return np.load(cache_file)[-n:]
        return np.load(cache_file)[:n]

    def __getitem__(self, idx: int | Iterable[int]) -> ContrastiveTriplet:
        if not self._is_loaded:
            raise RuntimeError("Call load() before using this method.")
        match self._status:
            case TCGADataStatus.SELFSUPERVISED:
                idx_len = (
                    len(range(*idx.indices(self._data.shape[0]))) if isinstance(idx, slice) else
                    len(idx) if isinstance(idx, Iterable) else
                    None
                )
                random_idx = np.random.randint(0, len(self._data), size=idx_len)
                x = ContrastiveTriplet(
                    anchor=augment(self._data[idx], self.noise_anchor),
                    positive=augment(self._data[idx], self.noise_positive),
                    negative=augment(self._data[random_idx], self.noise_negative)
                )
            case TCGADataStatus.SUPERVISED:
                x = self._data[idx]
        return x


    def to(self, device: torch.device) -> None:
        self._data = self._data.to(device)
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

    def _load_icir_data_from_lair(self,  n):
        adata_combined = self._load_icir_adata_from_lair(n)
        adata_combined = self._process_icir_adata_metadata(adata_combined)
        return adata_combined

    def _load_icir_adata_from_lair(self, n):
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
            if self.only_gide is True and filename != "Gide.h5ad":
                continue
            adata = ad.read_h5ad(path, backed="r")
            adata = adata[:, :] if n == 0 else adata[:, :n]
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

    def train_test_split(self, ratio=0.8, seed=0):

        train_dataset = copy.deepcopy(self)
        test_dataset = copy.deepcopy(self)

        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(len(self._full_data))
        n_split = int(np.round(ratio * len(self._full_data)))

        train_dataset._full_data = train_dataset._full_data[indices[:n_split]]
        test_dataset._full_data = test_dataset._full_data[indices[n_split:]]

        train_dataset.set_data()
        test_dataset.set_data()

        return train_dataset, test_dataset


def augment(self, x: torch.Tensor, additive_noise_level: float) -> torch.Tensor:
    add_noise = torch.distributions.Normal(loc=0, scale=additive_noise_level).sample(x.shape).to(x.device)
    cancer_idx = (slice(None), 0) if x.ndim >= 2 else (0,)
    add_noise[cancer_idx] = 0
    x = x + add_noise
    return x
