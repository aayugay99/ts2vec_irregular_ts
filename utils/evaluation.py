import hydra
import glob

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from lightgbm import LGBMClassifier

from utils.encode import encode_data



def bootstrap_eval(X_train, X_test, y_train, y_test, n_runs=10):

    lgbm = LGBMClassifier(
        n_estimators=500,
        boosting_type="gbdt",
        subsample=0.5,
        subsample_freq=1,
        learning_rate=0.02,
        feature_fraction=0.75,
        max_depth=6,
        lambda_l1=1,
        lambda_l2=1,
        min_data_in_leaf=50,
        random_state=42,
        n_jobs=8,
        verbose=-1
    )

    N = len(y_train)
    inds = np.arange(N)

    scores = []
    for _ in tqdm(range(n_runs)):
        bootstrap_inds = np.random.choice(inds, size=N, replace=True)

        lgbm.fit(X_train[bootstrap_inds], y_train[bootstrap_inds])

        y_pred = lgbm.predict_proba(X_test)

        if y_pred.shape[1] == 2:
            scores.append({
                "ROC-AUC": roc_auc_score(y_test, y_pred[:, 1]),
                "PR-AUC": average_precision_score(y_test, y_pred[:, 1]),
                "Accuracy": accuracy_score(y_test, y_pred.argmax(axis=1)),
            })
        
        else:
            scores.append({
                "ROC-AUC": roc_auc_score(y_test, y_pred, average="macro", multi_class="ovr"),
                "PR-AUC": average_precision_score(y_test, y_pred, average="macro"),
                "Accuracy": accuracy_score(y_test, y_pred.argmax(axis=1))
            })


    return pd.DataFrame(scores)


def checkpoints_eval(train_ds, test_ds, config_path, config_name):
    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name)  

    lgbm = LGBMClassifier(
        n_estimators=500,
        boosting_type="gbdt",
        subsample=0.5,
        subsample_freq=1,
        learning_rate=0.02,
        feature_fraction=0.75,
        max_depth=6,
        lambda_l1=1,
        lambda_l2=1,
        min_data_in_leaf=50,
        random_state=42,
        n_jobs=8,
        verbose=-1
    )

    seq_encoder = hydra.utils.instantiate(cfg["seq_encoder"])

    paths = glob.glob(f'{cfg["path_to_folder"]}/*.pth')
    print(f"Found {len(paths)} checkpoints.")

    scores = []
    for path in tqdm(paths):
        seq_encoder.load_state_dict(torch.load(path))

        X_train, y_train = encode_data(seq_encoder, train_ds)
        X_test, y_test = encode_data(seq_encoder, test_ds)

        lgbm.fit(X_train, y_train)

        y_pred = lgbm.predict_proba(X_test)

        if y_pred.shape[1] == 2:
            scores.append({
                "ROC-AUC": roc_auc_score(y_test, y_pred[:, 1]),
                "PR-AUC": average_precision_score(y_test, y_pred[:, 1]),
                "Accuracy": accuracy_score(y_test, y_pred.argmax(axis=1)),
            })
        
        else:
            scores.append({
                "ROC-AUC": roc_auc_score(y_test, y_pred, average="macro", multi_class="ovr"),
                "PR-AUC": average_precision_score(y_test, y_pred, average="macro"),
                "Accuracy": accuracy_score(y_test, y_pred.argmax(axis=1))
            })


    return pd.DataFrame(scores)