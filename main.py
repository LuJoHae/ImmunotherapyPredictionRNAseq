#!/usr/bin/env python

import numpy as np
import time
import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import trange

from immunotherapypredictionrnaseq.io import RunResults, RunConfig, setup_save_path
from immunotherapypredictionrnaseq.tokenizer import TokenConfig
from immunotherapypredictionrnaseq.model import Model, ModelWithHead
from immunotherapypredictionrnaseq.encoder import EncoderConfig
from immunotherapypredictionrnaseq.data import TCGAData, collate_triplets, TCGADataStatus
from immunotherapypredictionrnaseq.loss import TripletLoss
from immunotherapypredictionrnaseq.utils import check_params_and_gradients


def finetune(run_config: RunConfig, seed):
    assert run_config.weight_loading_datetime != ""
    torch.set_float32_matmul_precision('high')
    print_init_logging(run_config)
    save_path = Path.cwd().joinpath("runs")
    models_save_path, run_results = setup_file_output(save_path=save_path, run_config=run_config)
    token_config = setup_token_config()
    model = setup_model(run_config=run_config, token_config=token_config)
    weights_path = sorted(list(save_path.joinpath(run_config.weight_loading_datetime, "models").iterdir()))[-1]
    logger.info("path of loaded weights: {}".format(weights_path))
    model.load_state_dict(torch.load(weights_path))
    for param in model.parameters():
        param.requires_grad = False
    model_with_head = torch.compile(ModelWithHead(base_model=model, input_dim=45, hidden_dim=16))
    tcga_train, tcga_test = setup_dataset(run_config=run_config, token_config=token_config)
    assert tcga_train._data.shape != tcga_test._data.shape
    optimizer = torch.optim.Adam(model_with_head.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=run_config.patience
    )
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    for epoch in trange(run_config.n_epochs, desc="Epochs"):

        start = time.perf_counter()
        run_results.epoch = epoch

        train_loop(model_with_head, optimizer, run_config, run_results, tcga_train, cross_entropy_loss, mode="finetune")

        test_loop(model_with_head, run_config, run_results, tcga_test, cross_entropy_loss, mode="finetune")

        scheduler.step(run_results.loss_test_mean)
        run_results.lr = optimizer.param_groups[0]["lr"]
        run_results.runtime = time.perf_counter() - start
        logger.info("Epoch {}/{} finished in {} seconds.".format(epoch, run_config.n_epochs, run_results.runtime))
        run_results.save_row()
        torch.save(model_with_head.state_dict(), models_save_path.joinpath("{epoch:04d}.pth".format(epoch=epoch)))
        if run_results.lr <= 1e-8:
            break


def pretrain(run_config: RunConfig, seed):
    torch.set_float32_matmul_precision('high')
    print_init_logging(run_config)
    models_save_path, run_results = setup_file_output(save_path=Path.cwd().joinpath("runs"), run_config=run_config)
    token_config = setup_token_config()
    model = setup_model(run_config=run_config, token_config=token_config)
    tcga_train, tcga_test = setup_dataset(run_config=run_config, token_config=token_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=run_config.patience
    )
    triplet_loss = TripletLoss()

    for epoch in trange(run_config.n_epochs, desc="Epochs"):

        start = time.perf_counter()
        run_results.epoch = epoch

        train_loop(model, optimizer, run_config, run_results, tcga_train, triplet_loss)

        test_loop(model, run_config, run_results, tcga_test, triplet_loss)

        scheduler.step(run_results.loss_test_mean)
        run_results.lr = optimizer.param_groups[0]["lr"]
        run_results.runtime = time.perf_counter() - start
        logger.info("Epoch {}/{} finished in {} seconds.".format(epoch, run_config.n_epochs, run_results.runtime))
        run_results.save_row()
        torch.save(model.state_dict(), models_save_path.joinpath("{epoch:04d}.pth".format(epoch=epoch)))
        if run_results.lr <= 1e-8:
            break


def setup_token_config():
    config_path = Path.cwd().joinpath("token_config")
    assert config_path.exists() and config_path.is_dir()
    token_config = TokenConfig(config_path)
    token_config.load_config()
    return token_config


def setup_model(run_config, token_config):

    encoder_config = EncoderConfig(
        input_dim=len(token_config.genes),
        transformer_dim=run_config.transformer_dim,
        transformer_nhead=run_config.transformer_nhead,
        transformer_num_layers=run_config.transformer_num_layers,
        encoder_dropout=run_config.encoder_dropout
    )
    model = Model(encoder_config, token_config)
    model = model.to(run_config.device).to(torch.float32)
    logger.info("Start compiling model...")
    start = time.perf_counter()
    model = torch.compile(model)
    end = time.perf_counter()
    logger.info("Finished compiling model in {} seconds.".format(end - start))
    return model


def setup_dataset(run_config, token_config):
    logger.info("Start setting up dataset...")
    data = TCGAData(run_config.lair_path, token_config)
    data.set_status(status=TCGADataStatus.SUPERVISED if run_config.mode == "finetune" else TCGADataStatus.SELFSUPERVISED)
    logger.info("Dataset status set to {}.".format(data._status))
    if run_config.only_gide is True:
        data.only_gide = True
        logger.info("Only GIDE samples are used.")
    data.load(n=run_config.n_samples, cache=Path.cwd().joinpath("cache"))
    data.set_data()
    data.to(torch.device(run_config.device))
    train, test = data.train_test_split()
    return train, test


def print_init_logging(run_config):
    logger.setLevel(logging.INFO)
    logger.info(run_config)
    logger.info("Torch version: {}".format(torch.__version__))


def setup_file_output(save_path: Path, run_config: RunConfig) -> tuple[Path, RunResults]:
    run_save_path = setup_save_path(save_path)
    run_results = RunResults(run_save_path.joinpath("run_results.csv"))
    run_results.init_csv()
    run_config.save(run_save_path.joinpath("run_config.json"))
    models_save_path = run_save_path.joinpath("models")
    models_save_path.mkdir(parents=False, exist_ok=False)
    return models_save_path, run_results


def test_loop(model, run_config, run_results, tcga_test, loss_function, mode="pretrain"):
    logger.info("Start testing...")
    start_total = time.perf_counter()
    assert_correct_dataset_status(mode, tcga_test._status)
    model.eval()
    with torch.no_grad():
        assert_correct_dataset_status(mode, tcga_test._status)
        match mode:
            case "pretrain":
                logger.info("Use dataloader for pretrain")
                data_loader = DataLoader(
                    tcga_test,
                    batch_size=run_config.batch_size,
                    collate_fn=collate_triplets,
                    shuffle=True
                )
            case "finetune":
                logger.info("Use dataloader for finetune")
                data_loader = DataLoader(
                    tcga_test,
                    batch_size=run_config.batch_size,
                    shuffle=True
                )
        losses = []
        number_of_batches = int(np.ceil(len(tcga_test) / run_config.batch_size))
        for i, data in enumerate(data_loader, start=1):
            start = time.perf_counter()
            match mode:
                case "pretrain":
                    y = data.apply(lambda x: model(x)[1])
                    loss = loss_function(y)  # triplet loss
                case "finetune":
                    y = model(data[:, 1:])
                    print(data[:, 0].round().long())
                    loss = loss_function(y, data[:, 0].round().long())
            losses.append(loss.item())
            end = time.perf_counter()
            logger.info("batch {}/{} finished in {} seconds.".format(i, number_of_batches, end - start))
        run_results.loss_test_mean = np.mean(losses)
        run_results.loss_test_std = np.std(losses)
        del losses
    end_total = time.perf_counter()
    logger.info("Finished testing in {} seconds.".format(end_total - start_total))


def train_loop(model, optimizer, run_config, run_results, tcga_train, loss_function, mode="pretrain"):
    logger.info("Start training...")
    start_total = time.perf_counter()
    model.train()
    assert_correct_dataset_status(mode, tcga_train._status)
    match mode:
        case "pretrain":
            logger.info("Use dataloader for pretrain")
            data_loader = DataLoader(
                tcga_train,
                batch_size=run_config.batch_size,
                collate_fn=collate_triplets,
                shuffle=True
            )
        case "finetune":
            logger.info("Use dataloader for finetune")
            data_loader = DataLoader(
                tcga_train,
                batch_size=run_config.batch_size,
                shuffle=True
            )
    losses = []
    number_of_batches = int(np.ceil(len(tcga_train) / run_config.batch_size))
    for i, data in enumerate(data_loader, start=1):
        logger.info("new batch {}/{}".format(i, number_of_batches))
        start = time.perf_counter()
        optimizer.zero_grad()
        match mode:
            case "pretrain":
                y = data.apply(lambda x: model(x)[1])
                loss = loss_function(y)     # triplet loss
            case "finetune":
                y = model(data[:, 1:])
                print(data[:, 0].round().long())
                loss = loss_function(y, data[:, 0].round().long())     # cross entropy loss
        losses.append(loss.item())
        loss.backward()
        check_params_and_gradients(model)
        optimizer.step()
        check_params_and_gradients(model)
        end = time.perf_counter()
        logger.info("batch {}/{} finished in {} seconds.".format(i, number_of_batches, end - start))
    run_results.loss_train_mean = np.mean(losses)
    run_results.loss_train_std = np.std(losses)
    del losses
    end_total = time.perf_counter()
    logger.info("Finished training in {} seconds.".format(end_total - start_total))


def assert_correct_dataset_status(mode, status):
    logger.info("Checking if dataset is in correct status...")
    logger.info("Dataset status: {}".format(status))
    logger.info("Mode: {}".format(mode))
    match mode:
        case "pretrain":
            assert status == TCGADataStatus.SELFSUPERVISED
        case "finetune":
            assert status == TCGADataStatus.SUPERVISED
        case _:
            raise ValueError("Invalid mode: {}".format(mode))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # minimum level to log
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    for n in range(1):
        run_config = RunConfig.from_args()
        match run_config.mode:
            case "pretrain":
                pretrain(run_config, seed=n)
            case "finetune":
                finetune(run_config, seed=n)
            case _:
                raise EnvironmentError("Invalid run mode: {}".format(run_config.mode))
