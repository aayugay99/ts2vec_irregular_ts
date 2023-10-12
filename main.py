import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd

import torch

from sklearn.model_selection import train_test_split

def calc_diff(x):
    x_sorted = x.sort_values("timestamp")
    x_sorted["time_delta"] = (
        x_sorted["timestamp"] - x_sorted["timestamp"].shift().fillna(method="backfill")
    ).astype("int64") // 1000000000
    return x_sorted

@hydra.main(version_base=None, config_path="./config/age/ts2vec", config_name="ts2vec_age")
def main(cfg: DictConfig):
    df = pd.read_parquet(cfg["path"])

    df = df.groupby("user_id").apply(lambda x: calc_diff(x)).reset_index(drop=True)

    preprocessor = instantiate(cfg["preprocessor"])

    data = preprocessor.fit_transform(df)

    val_size, test_size = cfg["split"]["val_size"], cfg["split"]["test_size"]

    train, val_test = train_test_split(data, test_size=test_size+val_size, random_state=42)
    val, _ = train_test_split(val_test, test_size=test_size/(test_size+val_size), random_state=42)

    train_ds = instantiate(cfg["dataset"], data=train)
    val_ds = instantiate(cfg["dataset"], data=val)

    datamodule = instantiate(cfg["datamodule"], train_data=train_ds, valid_data=val_ds)

    model = instantiate(cfg["model"])

    checkpoint = instantiate(cfg["checkpoint"])
    callbacks = [checkpoint]

    if cfg["early_stopping"] is not None:
        es_callback = instantiate(cfg["early_stopping"])
        callbacks.append(es_callback)
    
    if cfg["logger"] is not None:
        logger = instantiate(cfg["logger"])
        trainer = instantiate(cfg["trainer"], callbacks=callbacks, logger=logger)
    else:
        trainer = instantiate(cfg["trainer"], callbacks=callbacks)

    trainer.fit(model, datamodule)

    model.load_state_dict(torch.load(checkpoint.best_model_path)["state_dict"])
    torch.save(model.seq_encoder.state_dict(), f'{cfg["name"]}.pth')

if __name__ == "__main__":
    main()
