import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QLineEdit, QAction

app = QApplication(sys.argv)
w = QMainWindow()

def edit_legend():
    try:
        items = ["Profile 1", "Profile 2"]
        item, ok = QInputDialog.getItem(w, "Edit Legend", "Select profile to rename:", items, 0, False)
        if ok and item:
            idx = items.index(item)
            new_label, ok2 = QInputDialog.getText(w, "New Legend", "Enter new legend text:", QLineEdit.Normal, "old_name")
            print(new_label)
    except Exception as e:
        print("EXCEPTION:", e)

a = QAction("Test", w)
a.triggered.connect(edit_legend)
a.trigger()
