"""Shared fit/predict interface; train-only transforms and explicit PyTorch loop."""

import copy
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from ridge_lsm import fit_ridge
from asian_lsm import asian_basis
from longstaff_schwartz import basis
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


class OriginalQuadratic:
    """Use the basis functions already present in the user's project."""

    def design(self, x):
        if x.shape[1] == 2:
            return asian_basis(x[:, 0], x[:, 1], 1.0)
        return basis(x[:, 0], 1.0, degree=2)

    def fit(self, x, y):
        self.beta = np.linalg.lstsq(self.design(x), y, rcond=None)[0]
        return self

    def predict(self, x):
        return self.design(x) @ self.beta


class OriginalRidge:
    """Extend ridge_lsm.fit_ridge to two-dimensional polynomial features.

    alpha/n translates sklearn's sum-loss alpha to fit_ridge's mean-loss penalty.
    """

    def __init__(self, degree=2, alpha=1.0):
        self.degree = degree
        self.alpha = alpha

    def fit(self, x, y):
        self.poly = PolynomialFeatures(self.degree, include_bias=True)
        self.beta = fit_ridge(self.poly.fit_transform(x), y, self.alpha / len(y))
        return self

    def predict(self, x):
        return self.poly.transform(x) @ self.beta


class NeuralRegressor:
    def __init__(
        self, width=32, optimizer="adam", lr=0.003, weight_decay=1e-4, epochs=60, seed=0
    ):
        self.width = width
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed

    def fit(self, x, y, validation=None):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        if validation is None:
            order = rng.permutation(len(x))
            cut = max(1, int(0.85 * len(x)))
            ix, iv = order[:cut], order[cut:]
            xt, yt = x[ix], y[ix]
            xv, yv = x[iv], y[iv]
        else:
            xt, yt = x, y
            xv, yv = validation
        self.scaler = StandardScaler().fit(xt)
        self.ym = float(np.mean(yt))
        self.ys = max(float(np.std(yt)), 1e-6)
        X = torch.tensor(self.scaler.transform(xt), dtype=torch.float32)
        Y = torch.tensor((yt - self.ym) / self.ys, dtype=torch.float32).reshape(-1, 1)
        V = torch.tensor(self.scaler.transform(xv), dtype=torch.float32)
        W = torch.tensor((yv - self.ym) / self.ys, dtype=torch.float32).reshape(-1, 1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], self.width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.width, self.width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.width, 1),
        )
        if self.optimizer == "adam":
            opt = torch.optim.Adam(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            opt = torch.optim.SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        best = float("inf")
        stale = 0
        self.history = []
        checkpoint = copy.deepcopy(self.net.state_dict())
        for epoch in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(X))
            for start in range(0, len(X), 512):
                idx = order[start : start + 512]
                opt.zero_grad()
                loss = torch.mean((self.net(X[idx]) - Y[idx]) ** 2)
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                tr = float(torch.mean((self.net(X) - Y) ** 2)) * self.ys**2
                va = float(torch.mean((self.net(V) - W) ** 2)) * self.ys**2
            self.history.append(
                {"epoch": epoch + 1, "train_mse": tr, "validation_mse": va}
            )
            if va < best - 1e-6:
                best = va
                stale = 0
                checkpoint = copy.deepcopy(self.net.state_dict())
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= 10:
                break
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        return self

    def predict(self, x):
        import torch

        with torch.no_grad():
            X = torch.tensor(self.scaler.transform(x), dtype=torch.float32)
            return self.net(X).numpy().ravel() * self.ys + self.ym


def make_model(spec, seed=0):
    p = spec.copy()
    kind = p.pop("kind")
    if kind == "quadratic":
        return OriginalQuadratic()
    if kind == "ridge":
        degree = p.pop("degree", 2)
        alpha = p.pop("alpha", 1.0)
        return OriginalRidge(degree, alpha)
    if kind == "forest":
        return RandomForestRegressor(
            n_estimators=30,
            max_depth=p.get("depth", 6),
            min_samples_leaf=p.get("leaf", 40),
            n_jobs=1,
            random_state=seed,
        )
    if kind == "boost":
        return HistGradientBoostingRegressor(
            max_iter=p.get("iterations", 60),
            max_leaf_nodes=p.get("leaves", 15),
            l2_regularization=1.0,
            min_samples_leaf=40,
            early_stopping=False,
            random_state=seed,
        )
    if kind == "neural":
        return NeuralRegressor(seed=seed, **p)
    raise ValueError(kind)
