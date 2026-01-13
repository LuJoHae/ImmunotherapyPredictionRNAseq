import numpy as np
import time
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import random
import pytest

from immunotherapypredictionrnaseq.io import RunResults, RunConfig, setup_save_path
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.model import Model
from immunotherapypredictionrnaseq.encoder import EncoderConfig
from immunotherapypredictionrnaseq.data import TCGAData, collate_triplets
from immunotherapypredictionrnaseq.loss import TripletLoss
from immunotherapypredictionrnaseq.utils import check_params_and_gradients


def setup_model_and_data(device, transformer_dim, transformer_nhead, transformer_num_layers, encoder_dropout, lair_path, n_samples):
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()

    encoder_config = EncoderConfig(
        input_dim=len(token_config.genes),
        transformer_dim=transformer_dim,
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers,
        encoder_dropout=encoder_dropout
    )
    model = Model(encoder_config, token_config)

    tcga_data = TCGAData(lair_path, token_config)
    tcga_data.load(n=n_samples, cache=Path.cwd().joinpath("cache"))

    device = torch.device(device)
    tcga_data.to(device)
    model = model.to(device).to(torch.float32)

    tcga_train, tcga_test = tcga_data.train_test_split()

    return model, tcga_train, tcga_test



def model_run(sample_run_config, n_epochs):
    run_config_sample = sample_run_config()
    run_config = RunConfig(
        name="Automated_Test_Run",
        type="test",
        device="cpu",
        lr=run_config_sample["lr"],
        batch_size=run_config_sample["batch_size"],
        weight_decay=1e-6,
        n_epochs=n_epochs,
        patience=3,
        transformer_dim=run_config_sample["transformer_dim_per_head"] * run_config_sample["transformer_nhead"],
        transformer_nhead=run_config_sample["transformer_nhead"],
        transformer_num_layers=1,
        encoder_dropout=run_config_sample["encoder_dropout"],
        lair_path=Path.cwd().joinpath("lair"),
        n_samples=run_config_sample["n_samples"]
    )
    model, tcga_train, tcga_test = setup_model_and_data(
        device=run_config.device,
        transformer_dim=run_config.transformer_dim,
        transformer_nhead=run_config.transformer_nhead,
        transformer_num_layers=run_config.transformer_num_layers,
        encoder_dropout=run_config.encoder_dropout,
        lair_path=run_config.lair_path,
        n_samples=run_config.n_samples
    )
    run_save_path = setup_save_path(Path.cwd().joinpath("runs"))
    run_results = RunResults(run_save_path.joinpath("run_results.csv"))
    run_results.init_csv()
    run_config.save(run_save_path.joinpath("run_config.json"))
    models_save_path = run_save_path.joinpath("models")
    models_save_path.mkdir(parents=False, exist_ok=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=run_config.patience)
    triplet_loss = TripletLoss()
    for epoch in range(run_config.n_epochs):

        start = time.perf_counter()
        run_results.epoch = epoch

        model.train()
        data_loader = DataLoader(
            tcga_train,
            batch_size=run_config.batch_size,
            collate_fn=collate_triplets,
            shuffle=True,
        )

        losses = []
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()
            y = data.apply(lambda x: model(x)[1])
            loss = triplet_loss(y)
            losses.append(loss.item())
            loss.backward()
            check_params_and_gradients(model)
            optimizer.step()
            check_params_and_gradients(model)
        run_results.loss_train_mean = np.mean(losses)
        run_results.loss_train_std = np.std(losses)
        del losses

        model.eval()
        with torch.no_grad():
            data_loader = DataLoader(
                tcga_test,
                batch_size=run_config.batch_size,
                collate_fn=collate_triplets,
                shuffle=True
            )
            losses = []
            for data in data_loader:
                y = data.apply(lambda x: model(x)[1])
                loss = triplet_loss(y)
                losses.append(loss.item())
            run_results.loss_test_mean = np.mean(losses)
            run_results.loss_test_std = np.std(losses)
            del losses

        scheduler.step(run_results.loss_test_mean)
        run_results.lr = optimizer.param_groups[0]["lr"]
        run_results.runtime = time.perf_counter() - start
        run_results.save_row()
        torch.save(model.state_dict(), models_save_path.joinpath("{epoch:04d}.pth".format(epoch=epoch)))


def test_integration_model_run():
    run_config_space = {
        "lr": [1e-2, 1e-3, 1e-4],
        "batch_size": range(8, 32),
        "transformer_dim_per_head": range(1, 3),
        "transformer_nhead": range(1, 3),
        "encoder_dropout": [0.0, 0.1],
        "n_samples": range(20, 30)
    }
    random.seed(0)
    sample_run_config = lambda: {key: random.choice(values) for key, values in run_config_space.items()}

    for _ in range(10):
        model_run(sample_run_config, n_epochs=3)


@pytest.mark.slow
def test_integration_model_run_long():
    run_config_space = {
        "lr": [1e-2, 1e-3, 1e-4],
        "batch_size": range(8, 32),
        "transformer_dim_per_head": range(10, 13),
        "transformer_nhead": range(3, 5),
        "encoder_dropout": [0.0, 0.1],
        "n_samples": range(20, 30)
    }
    random.seed(0)
    sample_run_config = lambda: {key: random.choice(values) for key, values in run_config_space.items()}

    for _ in range(10):
        model_run(sample_run_config, n_epochs=30)

