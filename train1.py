import paddle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from paddle_tabnet.tab_model import TabNetClassifier

np.random.seed(0)


import os
import wget
from pathlib import Path
import shutil
import gzip

from data import train, train_indices, valid_indices, test_indices


unused_feat = []
target = "Covertype"

features = [ col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = []

cat_dims = []

clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.7, clip_value=2.,
    optimizer_fn=paddle.optimizer.Adam,
    optimizer_params=dict(learning_rate=2e-2),
    scheduler_params={
        "learning_rate": 2e-2, "end_lr":0, "power":0.9, "decay_steps":3000 * 18 - 2000},
    scheduler_fn=paddle.optimizer.lr.PolynomialDecay,
    warmup=True,
    epsilon=1e-15,
    # resume_model='output/best_model',
    # last_epoch=1552,
    # last_best_acc=0.95889,
    prtrained_model='output/best_model'
)


X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]

max_epochs = 3000

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)],
    eval_name=['valid'],
    max_epochs=max_epochs, patience=1000,
    batch_size=16384, virtual_batch_size=256
)