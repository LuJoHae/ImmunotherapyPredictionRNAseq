from pathlib import Path
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.data import TCGAData

if __name__ == "__main__":
    cache = Path.cwd().joinpath("cache")
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()
    if not (cache.exists() and cache.is_dir()):
        raise IOError("Cache directory does not exist.")
    if any(cache.iterdir()):
        for file in cache.iterdir():
            if file.is_file():
                file.unlink()
    tcga_data = TCGAData(Path.cwd().joinpath("lair"), token_config)
    tcga_data.load(n=0, cache=cache)
