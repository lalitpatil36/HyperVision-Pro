import re

file_path = "/Users/lalit/othrwork/lalitnewsoftware/spectral_gui_2026_07_16.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace the broken block
pattern = r"    def train_ai_model\(self\) -> None:.*?(?=    def )"
# Note: we use DOTALL to capture everything until the next `def `

new_train = r"""    def train_ai_model(self) -> None:
        window = self.get_active_window()
        if not window:
            QMessageBox.warning(self, "No Active Cube", "Please open a data cube first.")
            return
            
        if not window.selections:
            QMessageBox.warning(self, "No ROIs", "Please extract some points or ROIs to train the AI model.")
            return
            
        unique_classes = set(sel["class_label"] for sel in window.selections)
        if len(unique_classes) < 2:
            QMessageBox.warning(self, "Insufficient Classes", "Please extract at least 2 different classes to train a classifier.")
            return
            
        dialog = AITrainingDialog(self, len(window.selections), len(unique_classes))
        if not dialog.exec_():
            return
            
        selected_models = dialog.get_selected_models()
        if "Auto (Evaluate All & Pick Best)" in selected_models:
            selected_models = [
                "Random Forest Classifier (Ensemble)",
                "Support Vector Machine (SVM)",
                "Neural Network (MLP 1D)",
                "K-Nearest Neighbors (KNN)",
                "Gradient Boosting Classifier",
                "Decision Tree Classifier",
                "Logistic Regression",
                "Gaussian Naive Bayes"
            ]
        
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
            from sklearn.preprocessing import label_binarize
            import matplotlib.pyplot as plt
            import joblib
            import os
            from pathlib import Path
            
        except ImportError:
            QMessageBox.critical(self, "Missing Dependencies", "scikit-learn or joblib not found. Please install them to use this feature.")
            return

        progress = QProgressDialog("Augmenting spectral data...", "Cancel", 0, 100, self)
        progress.setWindowTitle("AI Model Training")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        X_raw = []
        y_raw = []
        for sel in window.selections:
            X_raw.append(sel["profile"])
            y_raw.append(sel["class_label"])
            
        X_aug = []
        y_aug = []
        num_aug = 50
        
        for idx in range(len(X_raw)):
            base_profile = X_raw[idx]
            label = y_raw[idx]
            
            X_aug.append(base_profile)
            y_aug.append(label)
            
            for _ in range(num_aug):
                noise = np.random.normal(0, 0.02 * np.mean(base_profile), size=base_profile.shape)
                X_aug.append(base_profile + noise)
                y_aug.append(label)
                
        progress.setValue(20)
        QApplication.processEvents()
        
        X = np.array(X_aug)
        y = np.array(y_aug)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classes_unique = np.unique(y)
        
        work_dir = Path.cwd()
        results = []
        
        for i, model_name in enumerate(selected_models):
            if progress.wasCanceled():
                break
                
            progress.setLabelText(f"Training {model_name}...")
            progress.setValue(20 + int((i / len(selected_models)) * 60))
            QApplication.processEvents()
            
            if "Random Forest" in model_name:
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
            elif "Support Vector" in model_name:
                clf = SVC(kernel='rbf', probability=True)
            elif "Neural Network" in model_name:
                clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
            elif "Gradient Boosting" in model_name:
                clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif "Decision Tree" in model_name:
                clf = DecisionTreeClassifier(random_state=42)
            elif "Logistic Regression" in model_name:
                clf = LogisticRegression(max_iter=1000)
            elif "Gaussian Naive Bayes" in model_name:
                clf = GaussianNB()
            else:
                clf = KNeighborsClassifier(n_neighbors=5)
                
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            
            short_name = model_name.split()[0].replace("-", "") + "_Model"
            model_path = work_dir / f"{short_name}.joblib"
            report_path = work_dir / f"{short_name}_Report.txt"
            cm_path = work_dir / f"{short_name}_ConfusionMatrix.png"
            roc_path = work_dir / f"{short_name}_ROC.png"
            
            joblib.dump(clf, model_path)
            
            with open(report_path, "w") as f:
                f.write(f"HyperVision Pro AI Training Report\n")
                f.write(f"==================================\n")
                f.write(f"Datacube: {window.name}\n")
                f.write(f"Model Algorithm: {model_name}\n")
                f.write(f"Extracted Base Samples: {len(X_raw)}\n")
                f.write(f"Total Augmented Training Samples: {len(X_train)}\n")
                f.write(f"Total Validation Samples: {len(X_test)}\n\n")
                f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n")
                f.write("Detailed Classification Report:\n")
                f.write(report)
                
            # Plot Confusion Matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred, labels=classes_unique)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_unique)
            disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
            plt.title(f"Confusion Matrix - {model_name}")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close(fig)
            
            # Plot ROC Curve
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)
                y_test_bin = label_binarize(y_test, classes=classes_unique)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                if len(classes_unique) == 2:
                    # Binary classification
                    y_prob_positive = y_prob[:, 1]
                    fpr, tpr, _ = roc_curve(y_test_bin[:, 0] if y_test_bin.shape[1] > 1 else y_test_bin, y_prob_positive)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                else:
                    # Multi-class One-vs-Rest ROC
                    for c_idx, c_label in enumerate(classes_unique):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], y_prob[:, c_idx])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {c_label} (AUC = {roc_auc:.2f})")
                
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f"ROC Curve - {model_name}")
                ax.legend(loc="lower right", prop={'size': 8})
                plt.tight_layout()
                plt.savefig(roc_path, dpi=150)
                plt.close(fig)
                
            results.append({
                'name': model_name,
                'acc': acc,
                'model_path': model_path,
                'report': report_path
            })
            
            self.log(f"Trained {model_name}: Accuracy {acc*100:.2f}%")
            
        progress.setValue(100)
        
        if not results:
            return
            
        results.sort(key=lambda x: x['acc'], reverse=True)
        best = results[0]
        
        msg = f"Training Complete! Evaluated {len(results)} model(s).\n\n"
        msg += f"<b>Best Model: {best['name']}</b>\n"
        msg += f"Accuracy: {best['acc']*100:.2f}%\n\n"
        
        msg += "<b>All Results:</b>\n"
        for r in results:
            msg += f"- {r['name']}: {r['acc']*100:.2f}%\n"
            
        msg += f"\nAssets saved to: {work_dir}\n(Includes .joblib, Reports, Confusion Matrices, and ROC Curves)"
        
        QMessageBox.information(self, "AI Training Results", msg)

"""
content = re.sub(pattern, new_train, content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(content)
