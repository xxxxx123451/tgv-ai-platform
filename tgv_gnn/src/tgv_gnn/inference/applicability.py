from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ApplicabilityDomain:
    """Regularized Mahalanobis applicability domain fitted on training embeddings."""

    quantile: float = 0.95
    regularization: float = 1e-6
    center_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    threshold_: float | None = None

    def fit(self, embeddings) -> "ApplicabilityDomain":
        x = np.asarray(embeddings, dtype=float)
        if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
            raise ValueError("适用域训练嵌入必须是至少 3 行的有限二维数组。")
        self.center_ = x.mean(axis=0)
        covariance = np.cov(x, rowvar=False, ddof=0)
        covariance = np.atleast_2d(covariance) + np.eye(x.shape[1]) * self.regularization
        self.precision_ = np.linalg.pinv(covariance)
        distances = self.score(x)
        self.threshold_ = float(np.quantile(distances, self.quantile))
        return self

    def score(self, embeddings) -> np.ndarray:
        if self.center_ is None or self.precision_ is None:
            raise RuntimeError("ApplicabilityDomain 尚未拟合。")
        x = np.asarray(embeddings, dtype=float)
        delta = x - self.center_
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, self.precision_, delta), 0.0))

    def predict_in_domain(self, embeddings) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("ApplicabilityDomain 尚未拟合。")
        return self.score(embeddings) <= self.threshold_
