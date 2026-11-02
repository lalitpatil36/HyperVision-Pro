# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import pytest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([""])

from gmail_auth_dialog import AuthManager, GmailSignInDialog, UserProfileDialog


def test_auth_manager_validation():
    mgr = AuthManager()
    
    # Test empty email
    with pytest.raises(ValueError, match="cannot be empty"):
        mgr.sign_in_with_gmail("")

    # Test invalid email format
    with pytest.raises(ValueError, match="Invalid email format"):
        mgr.sign_in_with_gmail("invalid@domain-with-no-dot")

    # Test valid Gmail
    user = mgr.sign_in_with_gmail("researcher.hsi@gmail.com", remember_me=False)
    assert user is not None
    assert user["email"] == "researcher.hsi@gmail.com"
    assert "researcher" in user["name"].lower()
    assert user["provider"] == "Google / Gmail"
    assert len(user["token"]) == 32
    assert mgr.is_authenticated() is True
    assert mgr.get_current_user()["email"] == "researcher.hsi@gmail.com"

    # Test auto-appending @gmail.com
    user2 = mgr.sign_in_with_gmail("johndoe", remember_me=False)
    assert user2["email"] == "johndoe@gmail.com"


def test_auth_session_persistence():
    with tempfile.TemporaryDirectory() as td:
        mgr = AuthManager()
        mgr.config_dir = Path(td)
        mgr.session_file = Path(td) / "auth_session.json"
        mgr._current_user = None

        assert mgr.is_authenticated() is False
        assert mgr.get_current_user() is None

        # Sign in with remember_me = True
        user = mgr.sign_in_with_gmail("lab_scientist@gmail.com", remember_me=True)
        assert mgr.session_file.exists()
        assert mgr.is_authenticated() is True

        # Simulate fresh app startup
        fresh_mgr = AuthManager()
        fresh_mgr.config_dir = Path(td)
        fresh_mgr.session_file = Path(td) / "auth_session.json"
        fresh_mgr._load_session()
        assert fresh_mgr.is_authenticated() is True
        assert fresh_mgr.get_current_user()["email"] == "lab_scientist@gmail.com"

        # Test sign out
        fresh_mgr.sign_out()
        assert fresh_mgr.is_authenticated() is False
        assert not fresh_mgr.session_file.exists()


def test_gmail_dialogs_instantiation():
    mgr = AuthManager()
    mgr.sign_in_with_gmail("gui_tester@gmail.com", remember_me=False)

    sign_in_dlg = GmailSignInDialog(auth_manager=mgr)
    assert sign_in_dlg is not None
    assert sign_in_dlg.btn_google_oauth is not None
    assert sign_in_dlg.txt_email is not None

    profile_dlg = UserProfileDialog(auth_manager=mgr)
    assert profile_dlg is not None
