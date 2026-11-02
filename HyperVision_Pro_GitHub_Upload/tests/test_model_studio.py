# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import numpy as np
import pytest

from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from spectral_library import SpectralLibrary, SpectralLibraryDialog, SpectralMatcherDialog
from preprocessing_studio import PreprocessingStudioDialog
from model_visualization_studio import ModelVisualizationStudioDialog, COLORMAP_LIST, LINE_PALETTES


def test_spectral_library_dialog_parent_viewer_fix():
    """Verify bug fix: SpectralLibraryDialog accepts parent_viewer without error."""
    lib = SpectralLibrary("TestFixLib")
    dlg = SpectralLibraryDialog(lib, parent_viewer=None, parent=None)
    assert dlg.windowTitle() != ""
    assert dlg.parent_viewer is None
    dlg.close()

    class MockViewer:
        def __init__(self):
            self.image_windows = []
        def get_active_window(self):
            return None

    mock = MockViewer()
    dlg2 = SpectralLibraryDialog(lib, parent_viewer=mock)
    assert dlg2.parent_viewer is mock
    dlg2.close()


def test_spectral_matcher_dialog_parent_viewer():
    """Verify SpectralMatcherDialog accepts parent_viewer and kwargs."""
    lib = SpectralLibrary("TestMatcherLib")
    dlg = SpectralMatcherDialog(lib, parent_viewer=None, parent=None)
    assert dlg.windowTitle() != ""
    dlg.close()


def test_preprocessing_studio_kwargs():
    """Verify PreprocessingStudioDialog accepts parent_viewer and kwargs."""
    wv = np.linspace(400, 900, 30)
    spec = np.random.rand(2, 30)
    dlg = PreprocessingStudioDialog(wavelengths=wv, spectra=spec, parent_viewer=None, parent=None)
    assert dlg.windowTitle() != ""
    dlg.close()


def test_expanded_17_models_training():
    """Test training and probability output of all 17 spectral models."""
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
        HistGradientBoostingClassifier, AdaBoostClassifier
    )
    from sklearn.svm import SVC, LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import label_binarize

    class PLSDA(BaseEstimator, ClassifierMixin):
        def __init__(self, n_components=2):
            self.n_components = n_components
            self.pls = None
            self.classes_ = None
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            Y = label_binarize(y, classes=self.classes_)
            if len(self.classes_) == 2:
                Y = np.column_stack((1 - Y, Y))
            self.pls = PLSRegression(n_components=min(self.n_components, X.shape[1], max(1, X.shape[0]-1)))
            self.pls.fit(X, Y)
            return self
        def predict(self, X):
            Y_pred = self.pls.predict(X)
            return self.classes_[np.argmax(Y_pred, axis=1)]
        def predict_proba(self, X):
            Y_pred = self.pls.predict(X)
            Y_pred = np.clip(Y_pred, 0, None)
            row_sums = Y_pred.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1.0
            return Y_pred / row_sums

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=10, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=10, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
        "Linear SVC": CalibratedClassifierCV(LinearSVC(dual=False, max_iter=500, random_state=42)),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "PLS-DA": PLSDA(n_components=2),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Ridge": CalibratedClassifierCV(RidgeClassifier(random_state=42)),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gaussian Process": GaussianProcessClassifier(random_state=42)
    }

    assert len(models) == 17, f"Expected 17 models, got {len(models)}"

    np.random.seed(42)
    X = np.random.rand(60, 8)
    y = np.array(['Class_A'] * 30 + ['Class_B'] * 30)

    for name, clf in models.items():
        clf.fit(X, y)
        preds = clf.predict(X[:5])
        assert len(preds) == 5, f"Failed prediction for {name}"
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X[:5])
            assert probs.shape == (5, 2), f"Failed predict_proba shape for {name}"
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3), f"Probabilities do not sum to 1 in {name}"


def test_model_visualization_studio_all_modes_and_styling():
    """Verify ModelVisualizationStudioDialog chart modes, themes, text and colormaps."""
    dlg = ModelVisualizationStudioDialog()
    assert dlg.windowTitle() != ""
    assert dlg.cmb_model.count() > 0

    dlg.rad_cm.setChecked(True)
    dlg.cmb_cmap.setCurrentText("Viridis")
    dlg.chk_invert_cmap.setChecked(True)
    dlg.refresh_plot()
    assert dlg.rad_cm.isChecked()

    dlg.rad_roc.setChecked(True)
    dlg.cmb_palette.setCurrentText("Spectral")
    dlg.spin_line_width.setValue(3.0)
    dlg.refresh_plot()
    assert dlg.rad_roc.isChecked()

    dlg.rad_pr.setChecked(True)
    dlg.refresh_plot()
    assert dlg.rad_pr.isChecked()

    dlg.rad_benchmark.setChecked(True)
    dlg.cmb_bench_metric.setCurrentText("Overall Accuracy (%)")
    dlg.refresh_plot()
    assert dlg.rad_benchmark.isChecked()

    dlg.rad_features.setChecked(True)
    dlg.refresh_plot()
    assert dlg.rad_features.isChecked()

    dlg.txt_title.setText("Custom Unit Test Title")
    dlg.spin_title_size.setValue(18)
    dlg.txt_xlabel.setText("Custom X Wavelength")
    dlg.txt_ylabel.setText("Custom Y Reflectance")
    dlg.cmb_theme.setCurrentText("Dark Executive")
    dlg.refresh_plot()

    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, "test_export.png")
        dlg.figure.savefig(img_path, dpi=150, bbox_inches='tight')
        assert os.path.exists(img_path)
        assert os.path.getsize(img_path) > 1000

    dlg.close()
