import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb


def train_linear(X_tr, y_tr):
    mdl_lin = LinearRegression()
    mdl_lin.fit(X_tr, y_tr)
    return mdl_lin

def train_svm(X_tr, y_tr, params):
    mdl_svm = SVR(C = params['C'])
    mdl_svm.fit(X_tr, np.ravel(y_tr))
    return mdl_svm

def train_lgb(X_tr, y_tr, params):
    val_perc = 0.75
    N = len(X_tr)
    Ntr = int(val_perc*N)

    train_data_lgb = lgb.Dataset(X_tr.iloc[:Ntr], y_tr.iloc[:Ntr])
    val_data_lgb = lgb.Dataset(X_tr.iloc[Ntr:], y_tr.iloc[Ntr:])
    mdl_gbm = lgb.train(params,
                        train_data_lgb,
                        num_boost_round=1000,
                        valid_sets = [val_data_lgb, train_data_lgb],
                        valid_names = ['VA', 'TR'],
                        verbose_eval = 100)
    return mdl_gbm