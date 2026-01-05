from pathlib import Path
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.data import TCGAData


if __name__ == "__main__":
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()
    tcga_data = TCGAData(Path.cwd().joinpath("lair"), token_config)
    tcga_data.load(n=0, cache=Path.cwd().joinpath("cache/tcga_data.npy"))
