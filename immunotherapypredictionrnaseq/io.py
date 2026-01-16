import csv
import json
import os
from dataclasses import asdict, MISSING
from datetime import datetime
from pathlib import Path
import argparse
import dotenv
from pydantic.dataclasses import dataclass
from typing import get_type_hints, get_origin, get_args, Optional, Literal


class RunResults:
    epoch: int = None
    loss_train_mean: float = None
    loss_train_std: float = None
    loss_test_mean: float = None
    loss_test_std: float = None
    lr: float = None
    runtime: float = None

    def __init__(self, filepath: Path | str):
        self.filepath = filepath

    def init_csv(self):
        with open(self.filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "loss_train_mean",
                "loss_train_std",
                "loss_test_mean",
                "loss_test_std",
                "lr",
                "runtime"
            ])

    def save_row(self):
        with open(self.filepath, "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.epoch,
                self.loss_train_mean,
                self.loss_train_std,
                self.loss_test_mean,
                self.loss_test_std,
                self.lr,
                self.runtime
            ])


@dataclass()
class RunConfig:
    name: str = "Unnamed Run"
    type: Literal["full", "test"] = "test"
    mode: Literal["pretrain", "finetune"] = "pretrain"
    device: Literal["cpu", "cuda", "cuda:0", "mps:0"] = "cpu"
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-6
    n_epochs: int = 100
    patience: int = 5
    transformer_dim: int = 32
    transformer_nhead: int = 2
    transformer_num_layers: int = 1
    encoder_dropout: float = 0.1
    lair_path: Path = Path.cwd().joinpath("lair")
    weight_loading_datetime: str = ""
    n_samples: int = 0
    only_gide: bool = False

    def save(self, filepath: Path | str):
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4, default=str)

    @classmethod
    def load(cls, filepath: Path | str):
        with open(filepath, "r") as f:
            return RunConfig(**json.load(f))

    @classmethod
    def from_args(cls):
        parser = parser_from_dataclass(cls)
        args = parser.parse_args()
        args = {k: v for k, v in vars(args).items() if v is not None}
        return cls(**args)


def parser_from_dataclass(dc_cls) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    hints = get_type_hints(dc_cls)
    dotenv.load_dotenv()

    for name, typ in hints.items():
        field = dc_cls.__dataclass_fields__[name]
        arg_name = f"--{name.replace('_', '-')}"

        default = os.getenv(name.upper())
        required = False
        if default is None:
            default = field.default
            default_factory = field.default_factory
            required = (
                default is MISSING
                and default_factory is MISSING
            )

        origin = get_origin(typ)

        # 1) Literal[...] → use choices
        if origin is Literal:
            choices = get_args(typ)
            parser.add_argument(
                arg_name,
                choices=choices,
                default=None if required else default,
                required=required,
                help=f"{name} (choices: {choices})",
            )

        # 2) Optional[T] → type is T
        else:
            py_type = typ
            if origin is Optional:
                py_type = get_args(typ)[0]

            # Special case: bool flags
            if py_type is bool:
                # very simple version: --flag / --no-flag not handled here
                parser.add_argument(
                    arg_name,
                    action="store_true" if default is False else "store_false",
                    help=f"{name} (bool flag)",
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=py_type,
                    default=None if required else default,
                    required=required,
                    help=f"{name} ({typ})",
                )

    return parser


def setup_save_path(dir_path: Path) -> Path:
    run_save_path = dir_path.joinpath(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_save_path.mkdir(parents=True, exist_ok=False)
    return run_save_path
