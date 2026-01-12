#!/usr/bin/env python

import numpy as np
import time
import logging
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import trange

from immunotherapypredictionrnaseq.io import RunResults, RunConfig, setup_save_path
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.model import Model
from immunotherapypredictionrnaseq.encoder import EncoderConfig
from immunotherapypredictionrnaseq.data import TCGAData, collate_triplets
from immunotherapypredictionrnaseq.loss import TripletLoss
from immunotherapypredictionrnaseq.utils import check_params_and_gradients


def setup_model_and_data(device, transformer_dim, transformer_nhead, transformer_num_layers, encoder_dropout, lair_path, n_samples, seed):
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

    tcga_train, tcga_test = random_split(tcga_data, (0.8, 0.2), generator=torch.Generator().manual_seed(seed))

    return model, tcga_train, tcga_test


def main(run_config: RunConfig, seed):
    logging.basicConfig(
        level=logging.INFO,  # minimum level to log
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(run_config)
    logger.info("Torch version: {}".format(torch.__version__))

    model, tcga_train, tcga_test = setup_model_and_data(
        device=run_config.device,
        transformer_dim=run_config.transformer_dim,
        transformer_nhead=run_config.transformer_nhead,
        transformer_num_layers=run_config.transformer_num_layers,
        encoder_dropout=run_config.encoder_dropout,
        lair_path=run_config.lair_path,
        n_samples=run_config.n_samples,
        seed=seed
    )

    run_save_path = setup_save_path()
    run_results = RunResults(run_save_path.joinpath("run_results.csv"))
    run_results.init_csv()
    run_config.save(run_save_path.joinpath("run_config.json"))
    models_save_path = run_save_path.joinpath("models")
    models_save_path.mkdir(parents=False, exist_ok=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=run_config.patience)
    triplet_loss = TripletLoss()

    for epoch in trange(run_config.n_epochs, desc="Epochs"):

        start = time.perf_counter()
        run_results.epoch = epoch

        model.train()
        data_loader = DataLoader(
            tcga_train,
            batch_size=run_config.batch_size,
            collate_fn=collate_triplets,
            shuffle=True
        )

        losses = []
        for i, data in enumerate(data_loader):
            logger.info("new batch {}/{}".format(i, len(tcga_train)/run_config.batch_size))
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


if __name__ == "__main__":
    for seed in range(10):
        main(RunConfig.from_args(), seed=seed)
