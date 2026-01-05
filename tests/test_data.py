from immunotherapypredictionrnaseq.data import TCGAData
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from pathlib import Path


def test_tcga_data():
    config_path = Path("~").expanduser().joinpath("ImmunotherapyPredictionRNAseq").joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()
    tcga_data = TCGAData("/fast-storage/lair", token_config)
    tcga_data.load(full=False)
    _ = tcga_data[0:3]


if __name__ == "__main__":
    test_tcga_data()