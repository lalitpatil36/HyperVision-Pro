import re

file_path = "/Users/lalit/othrwork/lalitnewsoftware/spectral_gui_2026_07_16.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Update AITrainingDialog model list
old_models = r"""        models = \[
            "Auto \(Evaluate All & Pick Best\)",
            "Random Forest Classifier \(Ensemble\)",
            "Support Vector Machine \(SVM\)",
            "Neural Network \(MLP 1D\)",
            "K-Nearest Neighbors \(KNN\)",
            "Gradient Boosting Classifier",
            "Decision Tree Classifier",
            "Logistic Regression",
            "Gaussian Naive Bayes"
        \]"""
new_models = r"""        models = [
            "Auto (Evaluate All & Pick Best)",
            "Random Forest Classifier (Ensemble)",
            "Support Vector Machine (SVM)",
            "Neural Network (MLP 1D)",
            "K-Nearest Neighbors (KNN)",
            "Gradient Boosting Classifier",
            "Decision Tree Classifier",
            "Logistic Regression",
            "Gaussian Naive Bayes",
            "PLS-DA (Partial Least Squares)"
        ]"""
content = re.sub(old_models, new_models, content)

old_auto_list = r"""            selected_models = \[
                "Random Forest Classifier \(Ensemble\)",
                "Support Vector Machine \(SVM\)",
                "Neural Network \(MLP 1D\)",
                "K-Nearest Neighbors \(KNN\)",
                "Gradient Boosting Classifier",
                "Decision Tree Classifier",
                "Logistic Regression",
                "Gaussian Naive Bayes"
            \]"""
new_auto_list = r"""            selected_models = [
                "Random Forest Classifier (Ensemble)",
                "Support Vector Machine (SVM)",
                "Neural Network (MLP 1D)",
                "K-Nearest Neighbors (KNN)",
                "Gradient Boosting Classifier",
                "Decision Tree Classifier",
                "Logistic Regression",
                "Gaussian Naive Bayes",
                "PLS-DA (Partial Least Squares)"
            ]"""
content = re.sub(old_auto_list, new_auto_list, content)

# 2. Ask for directory & Add PLSDA logic
old_try = r"""        try:
            import numpy as np"""

new_try = r"""        out_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save AI Models & Reports")
        if not out_dir:
            return
            
        try:
            import numpy as np"""
content = content.replace(old_try, new_try)

old_imports = r"""            from sklearn.model_selection import train_test_split"""
new_imports = r"""            from sklearn.model_selection import train_test_split
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class PLSDA(BaseEstimator, ClassifierMixin):
                def __init__(self, n_components=2):
                    self.n_components = n_components
                    self.pls = PLSRegression(n_components=self.n_components)
                    self.classes_ = None
                def fit(self, X, y):
                    self.classes_ = np.unique(y)
                    Y = label_binarize(y, classes=self.classes_)
                    if len(self.classes_) == 2:
                        Y = np.column_stack((1 - Y, Y))
                    self.pls.fit(X, Y)
                    return self
                def predict(self, X):
                    Y_pred = self.pls.predict(X)
                    return self.classes_[np.argmax(Y_pred, axis=1)]
                def predict_proba(self, X):
                    Y_pred = self.pls.predict(X)
                    Y_pred = np.clip(Y_pred, 0, 1)
                    row_sums = Y_pred.sum(axis=1)[:, np.newaxis]
                    row_sums[row_sums == 0] = 1
                    return Y_pred / row_sums
"""
content = content.replace(old_imports, new_imports)

# 3. Update work_dir
content = content.replace("work_dir = Path.cwd()", "work_dir = Path(out_dir)")

# 4. Add PLSDA to instantiation logic
old_inst = r"""            else:
                clf = KNeighborsClassifier\(n_neighbors=5\)"""
new_inst = r"""            elif "PLS-DA" in model_name:
                clf = PLSDA(n_components=min(5, X_train.shape[1], X_train.shape[0]-1))
            else:
                clf = KNeighborsClassifier(n_neighbors=5)"""
content = re.sub(old_inst, new_inst, content)


with open(file_path, "w") as f:
    f.write(content)
