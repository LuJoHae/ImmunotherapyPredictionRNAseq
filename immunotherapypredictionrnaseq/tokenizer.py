import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
from numpy.typing import NDArray


class TokenConfig:
    cancer_types: pd.Series
    genes: pd.Series
    broad_celltype_pathways: pd.Series
    genesets: pd.DataFrame
    genesets_genes_junction_table: pd.DataFrame
    geneset_to_genes: Dict[str, NDArray[np.str_]]
    geneset_to_gene_indices: Dict[str, NDArray[np.int64]]
    celltype_pathway_to_genesets: Dict[str, NDArray[np.str_]]
    celltype_pathway_to_geneset_indicies: Dict[str, NDArray[np.int64]]
    _config_directory: Path


    def __init__(self, config_directory: Path) -> None:
        self._config_directory = config_directory


    def load_config(self) -> None:
        self._read_config_from_file()
        self._update_lookup_dicts()

    def _update_lookup_dicts(self):
        self.geneset_to_genes = self._extract_geneset_to_genes()
        self.geneset_to_gene_indices = self._extract_geneset_to_gene_indicies()
        self.celltype_pathway_to_genesets = self._extract_celltype_pathway_to_genesets()
        self.celltype_pathway_to_geneset_indicies = self._extract_celltype_pathway_to_geneset_indicies()
        self.cancer_type_to_code = self._extract_cancer_type_to_code()

    def read_config_tsv_series(self, filename: str) -> pd.Series:
        df = self.read_config_tsv(filename)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 1, df.columns
        series = df.iloc[:, 0]
        assert isinstance(series, pd.Series), type(series)
        series = series.rename_axis(None)
        return series

    def read_config_tsv_dataframe(self, filename: str, columns: List[str]) -> pd.DataFrame:
        df = self.read_config_tsv(filename)
        assert isinstance(df, pd.DataFrame)
        assert len(columns) == df.shape[1], df.columns
        df.columns = columns
        return df

    def read_config_tsv(self, filename) -> pd.DataFrame:
        return pd.read_csv(self._config_directory.joinpath(filename), index_col=None, sep="\t", header=None)

    def _read_config_from_file(self) -> None:
        self.cancer_types = self.read_config_tsv_series("cancer_types.tsv")
        self.genes = self.read_config_tsv_series("genes.tsv")
        self.broad_celltype_pathways = self.read_config_tsv_series("broad_celltype_pathways.tsv")
        self.genesets = self.read_config_tsv_dataframe("genesets.tsv", ["geneset", "broad_celltype_pathway"])
        self.genesets_genes_junction_table = self.read_config_tsv_dataframe("genesets+genes.tsv", ["geneset", "gene"])

    def _extract_geneset_to_genes(self) -> Dict[str, NDArray[np.str_]]:
        return self.genesets_genes_junction_table.groupby("geneset")["gene"].apply(np.array).to_dict()


    def _extract_geneset_to_gene_indicies(self) -> Dict[str, NDArray[np.str_]]:
        lookup = self.genes.reset_index().set_index(0)["index"]
        return {geneset: np.vectorize(lookup.get)(genes) for geneset, genes in self.geneset_to_genes.items()}


    def _extract_celltype_pathway_to_genesets(self):
        return {
            celltype_path: self.genesets.loc[self.genesets["broad_celltype_pathway"]==celltype_path]["geneset"].values
            for celltype_path in self.broad_celltype_pathways.values
        }

    def _extract_celltype_pathway_to_geneset_indicies(self) -> Dict[str, NDArray[np.int64]]:
        lookup = self.genesets.reset_index().set_index("geneset")["index"]
        return {celltype_pathway: np.vectorize(lookup.get)(geneset) for celltype_pathway, geneset in self.celltype_pathway_to_genesets.items()}

    def _extract_cancer_type_to_code(self) -> Dict[str, np.int64]:
        return self.cancer_types.reset_index().set_index(0)["index"].to_dict()
