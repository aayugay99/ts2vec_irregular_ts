import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from lightgbm import LGBMClassifier


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
