"""
dir_train_only.py - Train-fit / test-transform Disparate Impact Remover.

Background
----------
AIF360's `DisparateImpactRemover.fit_transform(dataset)` is transductive:
it computes per-protected-group empirical CDFs and the median target CDF
from whatever dataset is passed in, then repairs that same dataset in
place. There is no separate `fit(train).transform(test)` API.

If `fit_transform` is called on the full dataset before train/test split,
test rows contribute to the CDFs and to the median target, and the test
rows themselves are then repaired using statistics that include their own
contribution. That is the feature-distribution leakage flagged by IEEE
Access Reviewer 1 on manuscript #Access-2026-19472.

This module implements Feldman et al. (2015) quantile-mapping DIR with
proper train/test separation:

    1. fit(train_bld): learn per-group empirical CDFs and the median
       target CDF from TRAINING data only.
    2. transform(test_bld) / transform_array(X_test): for each row, find
       its quantile rank within its protected group's TRAINING CDF and
       remap to the value at that rank in the TRAINING-derived target CDF.

Test data is never used to compute any statistic. No leakage.

Reference
---------
Feldman, M., Friedler, S., Moeller, J., Scheidegger, C.,
Venkatasubramanian, S. (2015). Certifying and removing disparate impact.
KDD '15.
"""

from __future__ import annotations

import numpy as np


class TrainOnlyDisparateImpactRemover:
    """
    Quantile-mapping DIR with train/test separation.

    Parameters
    ----------
    repair_level : float in [0, 1]
        Blend between original (0) and fully repaired (1) values.
    sensitive_attribute : str
        Name of the protected attribute. Must be present in BLD
        feature_names when fit() is called.
    n_quantiles : int
        Resolution of empirical CDFs. 101 is a good default.
    """

    def __init__(self, repair_level: float = 1.0,
                 sensitive_attribute: str = None,
                 n_quantiles: int = 101):
        if not (0.0 <= repair_level <= 1.0):
            raise ValueError("repair_level must be in [0, 1]")
        if sensitive_attribute is None:
            raise ValueError("sensitive_attribute is required")
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")

        self.repair_level = float(repair_level)
        self.sensitive_attribute = sensitive_attribute
        self.n_quantiles = int(n_quantiles)
        self._levels = np.linspace(0.0, 1.0, self.n_quantiles)

        self._feat_idx: int | None = None
        self._groups: np.ndarray | None = None
        self._group_cdfs: dict[int, dict[int, np.ndarray]] = {}
        self._target_cdfs: dict[int, np.ndarray] = {}
        self._n_features: int | None = None

    # ---- fit ---------------------------------------------------------------

    def fit(self, train_bld) -> "TrainOnlyDisparateImpactRemover":
        feat_names = list(train_bld.feature_names)
        if self.sensitive_attribute not in feat_names:
            raise ValueError(
                f"sensitive_attribute {self.sensitive_attribute!r} not in "
                f"feature_names: {feat_names}"
            )
        feat_idx = feat_names.index(self.sensitive_attribute)
        features = np.asarray(train_bld.features, dtype=np.float64)
        prot = features[:, feat_idx].astype(int)
        groups = np.unique(prot)
        n_features = features.shape[1]

        group_cdfs: dict[int, dict[int, np.ndarray]] = {}
        target_cdfs: dict[int, np.ndarray] = {}

        for j in range(n_features):
            if j == feat_idx:
                continue
            group_cdfs[j] = {}
            stacked = []
            for g in groups:
                mask = (prot == g)
                if mask.sum() == 0:
                    vals = features[:, j]
                elif mask.sum() == 1:
                    vals = np.repeat(features[mask, j], 2)
                else:
                    vals = features[mask, j]
                cdf = np.quantile(vals, self._levels)
                group_cdfs[j][int(g)] = cdf
                stacked.append(cdf)
            target_cdfs[j] = np.median(np.stack(stacked), axis=0)

        self._feat_idx = feat_idx
        self._groups = groups
        self._group_cdfs = group_cdfs
        self._target_cdfs = target_cdfs
        self._n_features = n_features
        return self

    # ---- transform (numpy path) -------------------------------------------

    def transform_array(self, X) -> np.ndarray:
        """
        Repair a raw feature array using the train-fitted CDFs.

        Use this on the hot inference path to avoid BinaryLabelDataset
        construction overhead in latency measurements.
        """
        if self._feat_idx is None:
            raise RuntimeError("fit(train_bld) must be called before transform")
        X = np.asarray(X, dtype=np.float64).copy()
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Feature count mismatch: fit on {self._n_features}, "
                f"transform on {X.shape[1]}"
            )
        prot = X[:, self._feat_idx].astype(int)
        levels = self._levels

        for j in range(self._n_features):
            if j == self._feat_idx:
                continue
            target_cdf = self._target_cdfs[j]
            for g in self._groups:
                g_int = int(g)
                mask = (prot == g_int)
                if not mask.any():
                    continue
                group_cdf = self._group_cdfs[j].get(g_int)
                if group_cdf is None:
                    # Unseen group at inference: leave values unchanged.
                    continue
                vals = X[mask, j]
                pos = np.searchsorted(group_cdf, vals, side="right")
                ranks = np.clip(pos / self.n_quantiles, 0.0, 1.0)
                repaired = np.interp(ranks, levels, target_cdf)
                X[mask, j] = (
                    (1.0 - self.repair_level) * vals
                    + self.repair_level * repaired
                )
        return X

    # ---- transform (BLD path) ---------------------------------------------

    def transform(self, bld):
        """Repair a BinaryLabelDataset; thin wrapper over transform_array."""
        out = bld.copy()
        out.features = self.transform_array(out.features)
        return out

    def fit_transform(self, train_bld):
        """Convenience: fit on train_bld and return its repaired copy."""
        return self.fit(train_bld).transform(train_bld)
