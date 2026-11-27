import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import proteopy as pr


def _make_adata(X):
    obs = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(X.shape[0])]},
        index=[f"s{i}" for i in range(X.shape[0])],
    )
    var = pd.DataFrame(
        {"protein_id": [f"p{j}" for j in range(X.shape[1])]},
        index=[f"p{j}" for j in range(X.shape[1])],
    )
    return AnnData(X=X, obs=obs, var=var)


@pytest.mark.parametrize("sparse_cls", [sparse.csr_matrix, sparse.csc_matrix])
def test_calculate_cv_sparse_matches_dense_zero_to_na_true(sparse_cls):
    X = np.array(
        [
            [1000.0, 1000.0, 2000.0],
            [1000.0, 0.0, 2200.0],
            [0.0, 0.0, 2400.0],
        ],
        dtype=float,
    )
    ad_dense = _make_adata(X.copy())
    ad_sparse = _make_adata(sparse_cls(X.copy()))

    pr.pp.calculate_cv(ad_dense, zero_to_na=True)
    pr.pp.calculate_cv(ad_sparse, zero_to_na=True)

    np.testing.assert_allclose(
        ad_sparse.var["cv_X"].to_numpy(),
        ad_dense.var["cv_X"].to_numpy(),
        equal_nan=True,
    )


def test_calculate_cv_sparse_integer_zero_to_na_true():
    X = np.array(
        [
            [1000, 1000],
            [1000, 0],
            [0, 0],
        ],
        dtype=int,
    )
    ad_dense = _make_adata(X.copy())
    ad_sparse = _make_adata(sparse.csr_matrix(X.copy()))

    pr.pp.calculate_cv(ad_dense, zero_to_na=True)
    pr.pp.calculate_cv(ad_sparse, zero_to_na=True)

    np.testing.assert_allclose(
        ad_sparse.var["cv_X"].to_numpy(),
        ad_dense.var["cv_X"].to_numpy(),
        equal_nan=True,
    )


def test_calculate_cv_sparse_matches_dense_zero_to_na_false():
    X = np.array(
        [
            [1000.0, 1000.0],
            [1000.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    ad_dense = _make_adata(X.copy())
    ad_sparse = _make_adata(sparse.csr_matrix(X.copy()))

    pr.pp.calculate_cv(ad_dense, zero_to_na=False)
    pr.pp.calculate_cv(ad_sparse, zero_to_na=False)

    np.testing.assert_allclose(
        ad_sparse.var["cv_X"].to_numpy(),
        ad_dense.var["cv_X"].to_numpy(),
        equal_nan=True,
    )


def test_calculate_cv_sparse_group_by_matches_dense():
    X = np.array(
        [
            [1000.0, 1000.0],
            [1000.0, 0.0],
            [0.0, 0.0],
            [2000.0, 2000.0],
            [2200.0, 0.0],
            [2400.0, 0.0],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "group": ["A", "A", "A", "B", "B", "B"],
        },
        index=[f"s{i}" for i in range(6)],
    )
    var = pd.DataFrame(
        {"protein_id": ["p0", "p1"]},
        index=["p0", "p1"],
    )
    ad_dense = AnnData(X=X.copy(), obs=obs.copy(), var=var.copy())
    ad_sparse = AnnData(
        X=sparse.csr_matrix(X.copy()), obs=obs.copy(), var=var.copy()
    )

    pr.pp.calculate_cv(ad_dense, group_by="group", zero_to_na=True)
    pr.pp.calculate_cv(ad_sparse, group_by="group", zero_to_na=True)

    pd.testing.assert_frame_equal(
        ad_sparse.varm["cv_by_group_X"],
        ad_dense.varm["cv_by_group_X"],
    )
