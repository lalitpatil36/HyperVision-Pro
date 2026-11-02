# -*- coding: utf-8 -*-
import numpy as np
import pytest
from preprocessing_studio import compute_continuum_removal

def test_continuum_removal():
    wv = np.linspace(400, 1000, 60)
    curve = np.ones(60) * 0.5
    dip_idx = np.argmin(np.abs(wv - 680))
    curve[dip_idx] = 0.2

    cont, cr, features = compute_continuum_removal(wv, curve)
    assert len(cont) == len(wv)
    assert len(cr) == len(wv)
    assert cr[dip_idx] < 0.6
    assert np.all(cont >= curve - 1e-6)
    assert len(features) >= 1
    assert abs(features[0]["wavelength"] - 680) < 15.0

def test_pca_chemometrics_math():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X = np.random.rand(30, 20)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_scaled)

    eigenvals = pca.explained_variance_
    t2 = np.sum((scores ** 2) / eigenvals, axis=1)
    assert len(t2) == 30
    assert np.all(t2 >= 0)

    recon = pca.inverse_transform(scores)
    q = np.sum((X_scaled - recon) ** 2, axis=1)
    assert len(q) == 30
    assert np.all(q >= 0)
