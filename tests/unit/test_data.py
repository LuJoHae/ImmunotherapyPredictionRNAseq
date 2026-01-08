from immunotherapypredictionrnaseq.data import TCGAData
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from pathlib import Path
from dotenv import load_dotenv
import os


def test_tcga_data():
    load_dotenv()
    lair_path = os.getenv("LAIR_PATH")
    assert lair_path is not None
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()
    tcga_data = TCGAData(lair_path, token_config)
    n = 10
    tcga_data.load(n=n, cache=Path("./cache"))
    _ = tcga_data[0:3]
