#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:06:53 2020.

@author: malchiodi
"""

from mulearn import FuzzyInductor
from mulearn.kernel import PrecomputedKernel
from mulearn.fuzzifier import LinearFuzzifier
from mulearn.optimization import GurobiSolver
import csv
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV


def get_kernel_and_solver(gram):
    eigvals = np.linalg.eigvals(gram)
    assert (sum([abs(e.imag) for e in eigvals]) < 1e-4)
    abs_neg_eigvals = [-l.real for l in eigvals if l < 0]
    adjustment = max(abs_neg_eigvals) if abs_neg_eigvals else 0

    kernel = PrecomputedKernel(gram)
    solver = GurobiSolver(adjustment=adjustment) if adjustment else GurobiSolver()

    return kernel, solver


def get_dataset(filename):
    with open(filename) as data_file:
        data = np.array(list(csv.reader(data_file)))

    n = len(data) - 1
    n = 100

    # ## Extract data names, membership values and Gram matrix

    names = np.array(data[0])[1:n + 1]
    mu = np.array([float(row[0]) for row in data[1:n + 1]])
    gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n + 1]]
                     for row in data[1:n + 1]])

    assert (len(names.shape) == 1)
    assert (len(mu.shape) == 1)
    assert (len(gram.shape) == 2)

    assert (names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])

    X = np.array([[x] for x in np.arange(n)])

    return X, gram, mu


data_file_name = 'data/data-tettamanzi-complete.csv'
X, gram, mu = get_dataset(data_file_name)

out_cv = KFold()
i = 1

k, solver = get_kernel_and_solver(gram)
fi = FuzzyInductor(k=k, solver=solver, fuzzifier=LinearFuzzifier())

inner_folds = 5
gs = GridSearchCV(fi, {'c': np.logspace(-1, 1, 7)},
                  verbose=0, cv=inner_folds,
                  error_score=np.nan, n_jobs=1,
                  pre_dispatch=10)

train_scores = []
test_scores = []
for train_idx, test_idx in out_cv.split(X):
    X_train = X[train_idx]
    X_test = X[test_idx]
    mu_train = mu[train_idx]
    mu_test = mu[test_idx]

    gs.fit(X_train, mu_train)
    print(f"fold {i}: best parameters: {gs.best_params_['c']}")
    train_score = gs.score(X_train, mu_train)
    train_scores.append(train_score)
    test_score = gs.score(X_test, mu_test)
    test_scores.append(test_score)
    print(f'fold {i}: train score {train_score:.2f}, test score {test_score:.2f}')
    i += 1

print(f'train error: average {np.mean(train_scores):.3f}, std {np.std(train_scores):.3f}')
print(f'test error: average {np.mean(test_scores):.3f}, std {np.std(test_scores):.3f}')

