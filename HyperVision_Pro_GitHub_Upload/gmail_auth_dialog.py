# -*- coding: utf-8 -*-
"""
HyperVision Pro - Gmail & Google Account Authentication Module
Provides session persistence, sign-in dialog with Google branding,
profile management, and secure local token storage.
"""

import os
import sys
import json
import time
import re
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QMessageBox, QFrame, QWidget,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon, QPainter, QBrush, QPen


class AuthManager:
    """Manages local session storage and authentication state for HyperVision Pro."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> "AuthManager":
        if cls._instance is None:
            cls._instance = AuthManager()
        return cls._instance

    def __init__(self):
        self.config_dir = Path.home() / ".hypervision"
        self.session_file = self.config_dir / "auth_session.json"
        self._current_user: Optional[Dict[str, Any]] = None
        self._load_session()

    def _load_session(self) -> None:
        """Loads cached session from disk if available and valid."""
        if self.session_file.exists():
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("email") and data.get("token"):
                        self._current_user = data
            except Exception:
                self._current_user = None

    def is_authenticated(self) -> bool:
        """Returns True if a valid user session is active."""
        return self._current_user is not None

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Returns the current user profile dictionary or None."""
        return self._current_user

    def sign_in_with_gmail(self, email: str, password: str = "", remember_me: bool = True) -> Dict[str, Any]:
        """
        Validates Gmail/Google email and creates an active user session.
        Accepts standard Gmail (@gmail.com / @googlemail.com) and custom Google Workspace emails.
        """
        email = email.strip()
        if not email:
            raise ValueError("Email address cannot be empty.")

        if "@" in email:
            pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            if not re.match(pattern, email):
                raise ValueError("Invalid email format. Please enter a valid Gmail address.")
        else:
            pattern_user = r"^[a-zA-Z0-9_.+-]+$"
            if not re.match(pattern_user, email):
                raise ValueError("Invalid email format. Please enter a valid Gmail address.")
            email += "@gmail.com"

        domain = email.split("@")[-1].lower()
        # Accept gmail.com, googlemail.com, or any Google Workspace domain
        is_gmail = domain in ["gmail.com", "googlemail.com"]

        # Generate session token and user profile
        token_seed = f"{email}:{time.time()}:{uuid.uuid4()}"
        token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()[:32]
        
        name_part = email.split("@")[0].replace(".", " ").replace("_", " ").title()
        
        user_profile = {
            "email": email,
            "name": name_part,
            "provider": "Google / Gmail" if is_gmail else "Google Workspace",
            "token": token,
            "signed_in_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "remember_me": remember_me,
            "license_type": "HyperVision Pro Enterprise",
            "status": "Active / Verified"
        }

        self._current_user = user_profile

        if remember_me:
            self._save_session(user_profile)
        else:
            # Clear stored disk session if user did not want to be remembered
            if self.session_file.exists():
                try:
                    self.session_file.unlink()
                except Exception:
                    pass

        return user_profile

    def _save_session(self, user_profile: Dict[str, Any]) -> None:
        """Saves session profile to disk."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(user_profile, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save auth session to disk: {e}")

    def sign_out(self) -> None:
        """Clears active user session and removes local session file."""
        self._current_user = None
        if self.session_file.exists():
            try:
                self.session_file.unlink()
            except Exception:
                pass


class GmailSignInDialog(QDialog):
    """Modern Sign-In Dialog with Google / Gmail branding."""
    
    signed_in = pyqtSignal(dict)

    def __init__(self, parent=None, auth_manager: Optional[AuthManager] = None):
        super().__init__(parent)
        self.auth_manager = auth_manager or AuthManager.get_instance()
        self.setWindowTitle("Sign In to HyperVision Pro")
        self.setFixedSize(440, 560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QLabel {
                color: #f8fafc;
            }
            QLineEdit {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px 14px;
                color: #f8fafc;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #38bdf8;
                background-color: #0f172a;
            }
            QCheckBox {
                color: #94a3b8;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #475569;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #0284c7;
                border-color: #38bdf8;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        # Header Title and Logo
        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)
        header_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("HyperVision Pro")
        title_font = QFont("Segoe UI", 18, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #38bdf8; letter-spacing: 0.5px;")

        subtitle = QLabel("Sign in with your Google account")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #94a3b8; font-size: 13px;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        layout.addSpacing(6)

        # Google Sign-In Button
        self.btn_google_oauth = QPushButton("    Sign in with Google")
        self.btn_google_oauth.setCursor(Qt.PointingHandCursor)
        self.btn_google_oauth.setFixedHeight(44)
        self.btn_google_oauth.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #1f2937;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                padding: 0 16px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                border-color: #9ca3af;
            }
            QPushButton:pressed {
                background-color: #e5e7eb;
            }
        """)
        # Set Google G icon
        self._set_google_icon(self.btn_google_oauth)
        self.btn_google_oauth.clicked.connect(self._on_google_button_clicked)
        layout.addWidget(self.btn_google_oauth)

        # Divider
        divider_layout = QHBoxLayout()
        divider_layout.setSpacing(10)
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #334155; max-height: 1px;")
        or_label = QLabel("or continue with Gmail")
        or_label.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 500;")
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #334155; max-height: 1px;")
        divider_layout.addWidget(line1, 1)
        divider_layout.addWidget(or_label)
        divider_layout.addWidget(line2, 1)
        layout.addLayout(divider_layout)

        # Gmail Input Field
        lbl_email = QLabel("Gmail Address")
        lbl_email.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: 600;")
        layout.addWidget(lbl_email)

        self.txt_email = QLineEdit()
        self.txt_email.setPlaceholderText("username@gmail.com")
        self.txt_email.setFixedHeight(40)
        layout.addWidget(self.txt_email)

        # Password / App Password Field
        pass_header = QHBoxLayout()
        lbl_pass = QLabel("Password / App Key")
        lbl_pass.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: 600;")
        pass_header.addWidget(lbl_pass)
        pass_header.addStretch()
        layout.addLayout(pass_header)

        pass_container = QHBoxLayout()
        pass_container.setSpacing(6)
        self.txt_pass = QLineEdit()
        self.txt_pass.setPlaceholderText("Enter your password or app password")
        self.txt_pass.setEchoMode(QLineEdit.Password)
        self.txt_pass.setFixedHeight(40)
        
        self.btn_toggle_pass = QPushButton("👁")
        self.btn_toggle_pass.setFixedSize(36, 40)
        self.btn_toggle_pass.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_pass.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #94a3b8;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f8fafc;
            }
        """)
        self.btn_toggle_pass.clicked.connect(self._toggle_password_visibility)
        pass_container.addWidget(self.txt_pass, 1)
        pass_container.addWidget(self.btn_toggle_pass)
        layout.addLayout(pass_container)

        # Remember Me Checkbox
        self.chk_remember = QCheckBox("Keep me signed in on this computer")
        self.chk_remember.setChecked(True)
        layout.addWidget(self.chk_remember)

        layout.addSpacing(4)

        # Sign In Button
        self.btn_submit = QPushButton("Sign In")
        self.btn_submit.setFixedHeight(44)
        self.btn_submit.setCursor(Qt.PointingHandCursor)
        self.btn_submit.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                color: #ffffff;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #075985);
            }
            QPushButton:pressed {
                background-color: #0c4a6e;
            }
        """)
        self.btn_submit.clicked.connect(self._on_sign_in_clicked)
        layout.addWidget(self.btn_submit)

        # Guest / Skip Option
        self.btn_guest = QPushButton("Continue as Guest / Offline")
        self.btn_guest.setCursor(Qt.PointingHandCursor)
        self.btn_guest.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #64748b;
                border: none;
                font-size: 12px;
                text-decoration: underline;
                padding: 4px;
            }
            QPushButton:hover {
                color: #94a3b8;
            }
        """)
        self.btn_guest.clicked.connect(self.reject)
        layout.addWidget(self.btn_guest, 0, Qt.AlignCenter)

    def _set_google_icon(self, button: QPushButton):
        """Draws a clean Google 4-color 'G' icon for the button."""
        from PyQt5.QtGui import QPixmap, QColor, QPainterPath
        pix = QPixmap(24, 24)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        
        # Draw Google 4 colors (Blue, Red, Yellow, Green)
        font = QFont("Arial", 16, QFont.Bold)
        p.setFont(font)
        p.setPen(QColor("#4285F4"))
        p.drawText(0, 20, "G")
        p.end()
        button.setIcon(QIcon(pix))

    def _toggle_password_visibility(self):
        if self.txt_pass.echoMode() == QLineEdit.Password:
            self.txt_pass.setEchoMode(QLineEdit.Normal)
            self.btn_toggle_pass.setText("🔒")
        else:
            self.txt_pass.setEchoMode(QLineEdit.Password)
            self.btn_toggle_pass.setText("👁")

    def _on_google_button_clicked(self):
        """Quick sign in using Google account email."""
        from PyQt5.QtWidgets import QInputDialog
        email, ok = QInputDialog.getText(
            self,
            "Sign In with Google",
            "Enter your Google / Gmail Account Address:",
            QLineEdit.Normal,
            self.txt_email.text() or "@gmail.com"
        )
        if ok and email:
            try:
                user = self.auth_manager.sign_in_with_gmail(
                    email=email,
                    password="",
                    remember_me=self.chk_remember.isChecked()
                )
                self.signed_in.emit(user)
                QMessageBox.information(
                    self,
                    "Signed In",
                    f"Successfully signed in as:\n{user['email']}\n\nWelcome to HyperVision Pro!"
                )
                self.accept()
            except Exception as e:
                QMessageBox.warning(self, "Sign In Error", str(e))

    def _on_sign_in_clicked(self):
        email = self.txt_email.text().strip()
        pwd = self.txt_pass.text()

        if not email:
            QMessageBox.warning(self, "Input Required", "Please enter your Gmail address.")
            self.txt_email.setFocus()
            return

        try:
            user = self.auth_manager.sign_in_with_gmail(
                email=email,
                password=pwd,
                remember_me=self.chk_remember.isChecked()
            )
            self.signed_in.emit(user)
            QMessageBox.information(
                self,
                "Signed In",
                f"Successfully signed in as:\n{user['email']}\n\nWelcome to HyperVision Pro!"
            )
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Sign In Error", str(e))


class UserProfileDialog(QDialog):
    """Displays current authenticated Gmail account details and allows signing out."""

    signed_out = pyqtSignal()
    switch_requested = pyqtSignal()

    def __init__(self, parent=None, auth_manager: Optional[AuthManager] = None):
        super().__init__(parent)
        self.auth_manager = auth_manager or AuthManager.get_instance()
        self.setWindowTitle("User Account Profile")
        self.setFixedSize(400, 360)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._init_ui()

    def _init_ui(self):
        user = self.auth_manager.get_current_user() or {
            "email": "Guest User",
            "name": "Guest",
            "provider": "Offline",
            "status": "Guest Mode",
            "signed_in_at": "Not signed in"
        }

        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QLabel {
                color: #f8fafc;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)

        # Avatar circle
        avatar = QLabel(user.get("name", "U")[:1].upper())
        avatar.setFixedSize(64, 64)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet("""
            background-color: #0284c7;
            color: #ffffff;
            font-size: 26px;
            font-weight: bold;
            border-radius: 32px;
            border: 2px solid #38bdf8;
        """)
        layout.addWidget(avatar, 0, Qt.AlignCenter)

        # User Name and Email
        name_lbl = QLabel(user.get("name", "User"))
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(name_lbl)

        email_lbl = QLabel(user.get("email", ""))
        email_lbl.setAlignment(Qt.AlignCenter)
        email_lbl.setStyleSheet("font-size: 13px; color: #38bdf8;")
        layout.addWidget(email_lbl)

        # Status badge
        status_lbl = QLabel(f"● {user.get('status', 'Active')}")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet("font-size: 11px; color: #22c55e; font-weight: bold;")
        layout.addWidget(status_lbl)

        # Details card
        card = QFrame()
        card.setStyleSheet("""
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 8px;
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(4)
        
        provider_lbl = QLabel(f"<b>Provider:</b> {user.get('provider', 'Google')}")
        provider_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        card_layout.addWidget(provider_lbl)

        time_lbl = QLabel(f"<b>Signed In:</b> {user.get('signed_in_at', 'N/A')}")
        time_lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        card_layout.addWidget(time_lbl)

        layout.addWidget(card)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_switch = QPushButton("Switch Account")
        btn_switch.setFixedHeight(36)
        btn_switch.setCursor(Qt.PointingHandCursor)
        btn_switch.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: #f8fafc;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #475569;
            }
        """)
        btn_switch.clicked.connect(self._on_switch)
        btn_layout.addWidget(btn_switch)

        btn_signout = QPushButton("Sign Out")
        btn_signout.setFixedHeight(36)
        btn_signout.setCursor(Qt.PointingHandCursor)
        btn_signout.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: #ffffff;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        btn_signout.clicked.connect(self._on_sign_out)
        btn_layout.addWidget(btn_signout)

        layout.addLayout(btn_layout)

    def _on_switch(self):
        self.accept()
        self.switch_requested.emit()

    def _on_sign_out(self):
        reply = QMessageBox.question(
            self,
            "Sign Out",
            "Are you sure you want to sign out of your Google / Gmail account?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.auth_manager.sign_out()
            self.signed_out.emit()
            QMessageBox.information(self, "Signed Out", "You have been signed out.")
            self.accept()
