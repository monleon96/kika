"""
Authentication helpers for the Streamlit UI.

Provides email/password registration, login, guest access, and session
management. User credentials are stored securely (bcrypt hashes) in the
local SQLite database so the system can scale beyond a single user without
changing much code when deploying.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from passlib.context import CryptContext

from .db import init_db, db_cursor
from .email_service import EmailServiceError, send_email
from .token_service import (
    DEFAULT_EXPIRY_SECONDS,
    VerificationTokenExpired,
    VerificationTokenInvalid,
    confirm_email_verification_token,
    generate_email_verification_token,
    get_public_base_url,
)
from .user_settings import (
    USER_SETTINGS_KEY,
    USER_SETTINGS_OWNER_KEY,
    bootstrap_user_settings,
    get_default_settings,
)


AUTH_SESSION_KEY = "authenticated_user"
_AUTH_INITIALISED = False
_VERIFICATION_FEEDBACK_KEY = "email_verification_feedback"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PASSWORD_POLICY = "Password must be between 8 and 128 characters long."
_PWD_CONTEXT = CryptContext(
    schemes=["bcrypt_sha256", "bcrypt"],
    deprecated="auto",
    bcrypt_sha256__truncate_error=False,
    bcrypt__truncate_error=False,
)


def _ensure_auth_tables() -> None:
    global _AUTH_INITIALISED
    if not _AUTH_INITIALISED:
        init_db()
        _AUTH_INITIALISED = True


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str) -> str:
    password = str(password)  # normalize to string
    return _PWD_CONTEXT.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    return _PWD_CONTEXT.verify(str(password), password_hash)


def _make_user_dict(row: Any, *, is_guest: bool = False) -> Dict[str, Any]:
    if is_guest:
        return {"id": None, "email": None, "full_name": "Guest", "is_guest": True}
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "is_guest": False,
    }


def register_user(full_name: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Create a new user account.
    """
    _ensure_auth_tables()

    full_name = full_name.strip()
    if not full_name:
        return False, "Please provide your name."

    email = _normalize_email(email)
    if not email or not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."

    if len(password) < 8 or len(password) > 128:
        return False, _PASSWORD_POLICY

    try:
        token = generate_email_verification_token(email)
    except RuntimeError as exc:
        return False, str(exc)

    try:
        with db_cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return False, "An account with this email already exists."

            cursor.execute(
                """
                INSERT INTO users (email, full_name, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (email, full_name, _hash_password(password), datetime.utcnow().isoformat(timespec="seconds")),
            )

            sent, delivery_message = send_verification_email(email, full_name, token=token)
            if not sent:
                raise RuntimeError(delivery_message)
    except RuntimeError as exc:
        return False, str(exc)

    return True, "Account created successfully. Check your inbox to verify your email."


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Validate credentials and return a user dictionary on success.
    """
    _ensure_auth_tables()

    email = _normalize_email(email)
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, full_name, password_hash, email_verified
            FROM users
            WHERE email = ? AND is_active = 1
            """,
            (email,),
        )
        row = cursor.fetchone()

    if not row:
        return False, None, "Invalid email or password."

    if not _verify_password(password, row["password_hash"]):
        return False, None, "Invalid email or password."

    if not row["email_verified"]:
        return False, None, "Email address not verified. Please check your inbox or request a new verification link."

    with db_cursor() as cursor:
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow().isoformat(timespec="seconds"), row["id"]),
        )

    return True, _make_user_dict(row), "Signed in successfully."


def logout_user() -> None:
    """
    Clear session state for the current user.
    """
    for key in (AUTH_SESSION_KEY, USER_SETTINGS_KEY, USER_SETTINGS_OWNER_KEY):
        if key in st.session_state:
            del st.session_state[key]

    for legacy_key in (
        "njoy_exe_path",
        "njoy_version",
        "njoy_output_dir",
        "njoy_create_xsdir",
        "njoy_auto_version",
    ):
        if legacy_key in st.session_state:
            del st.session_state[legacy_key]


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Return the authenticated user dictionary (or None).
    """
    return st.session_state.get(AUTH_SESSION_KEY)


def require_user() -> Dict[str, Any]:
    """
    Guarantee an authenticated user is present, rendering the auth portal if required.
    """
    _ensure_auth_tables()

    user = get_current_user()
    if user:
        bootstrap_user_settings(user.get("id"), user.get("is_guest", False))
        return user

    _render_auth_portal()
    st.stop()


def _render_auth_portal() -> None:
    st.title("🔐 Welcome to KIKA")
    st.markdown("Sign in to access your projects, or continue as a guest to explore with default settings.")

    tab_login, tab_register, tab_guest = st.tabs(["Sign In", "Create Account", "Guest"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", key="auth_login_email")
            password = st.text_input("Password", type="password", key="auth_login_password")
            submitted = st.form_submit_button("Sign in", type="primary")

        if submitted:
            success, user, message = authenticate_user(email, password)
            if success and user:
                st.session_state[AUTH_SESSION_KEY] = user
                bootstrap_user_settings(user.get("id"), False)
                st.success(message)
                st.rerun()
            else:
                if "not verified" in message.lower():
                    st.warning(message)
                    st.session_state["auth_pending_email"] = _normalize_email(email)
                else:
                    st.error(message)

        with st.expander("Need a new verification email?", expanded=False):
            default_resend_email = st.session_state.get("auth_pending_email", "")
            with st.form("resend_verification_form"):
                resend_email = st.text_input(
                    "Email address",
                    value=default_resend_email,
                    key="auth_resend_email_input",
                )
                resend_submit = st.form_submit_button("Resend verification link")
            if resend_submit:
                sent, resend_message = resend_verification_email(resend_email)
                st.session_state["auth_pending_email"] = _normalize_email(resend_email)
                if sent:
                    st.success(resend_message)
                else:
                    st.warning(resend_message)

    with tab_register:
        with st.form("register_form"):
            full_name = st.text_input("Full name", key="auth_register_full_name")
            email = st.text_input("Email address", key="auth_register_email")
            password = st.text_input("Password", type="password", key="auth_register_password")
            confirm_password = st.text_input("Confirm password", type="password", key="auth_register_password_confirm")
            submitted = st.form_submit_button("Create account", type="primary")

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(full_name, email, password)
                if success:
                    st.success("Account created! Verify your email to continue.")
                    st.info(message)
                    st.session_state["auth_pending_email"] = _normalize_email(email)
                    st.info("Once verified, return to the Sign In tab.")
                else:
                    st.error(message)

    with tab_guest:
        st.info(
            "Continue as a guest to try KIKA without creating an account. "
            "Your settings will reset after you leave."
        )
        if st.button("Continue as guest", type="secondary"):
            guest_user = _make_user_dict(
                {"id": None, "email": None, "full_name": "Guest"},
                is_guest=True,
            )
            st.session_state[AUTH_SESSION_KEY] = guest_user
            st.session_state[USER_SETTINGS_KEY] = get_default_settings()
            st.session_state[USER_SETTINGS_OWNER_KEY] = "guest"
            bootstrap_user_settings(None, True)
            st.query_params.clear()  # Clear any lingering parameters
            st.success("Continuing as guest.")
            st.rerun()


def send_verification_email(email: str, full_name: str, *, token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Send an email verification link to the user.
    """
    email = _normalize_email(email)
    full_name = full_name.strip() or "there"
    try:
        token = token or generate_email_verification_token(email)
    except RuntimeError as exc:
        return False, str(exc)

    verify_url = f"{get_public_base_url()}/?verify={token}"
    hours = DEFAULT_EXPIRY_SECONDS // 3600
    subject = "Verify your KIKA account"
    body = (
        f"Hi {full_name},\n\n"
        "Please verify your KIKA account by following the link below:\n\n"
        f"{verify_url}\n\n"
        f"This link expires in {hours} hours. If you did not create this account, "
        "you can safely ignore this email.\n\n"
        "— The KIKA Team"
    )
    html_body = (
        f"<p>Hi {full_name},</p>"
        "<p>Please verify your <strong>KIKA</strong> account by clicking the link below:</p>"
        f"<p><a href=\"{verify_url}\">{verify_url}</a></p>"
        f"<p>This link expires in {hours} hours. If you did not create this account, "
        "you can ignore this email.</p>"
        "<p>— The KIKA Team</p>"
    )

    try:
        sent = send_email(email, subject, body, html=html_body)
    except EmailServiceError as exc:
        return False, f"Unable to send verification email: {exc}"

    if not sent:
        return False, "Email service is not configured (set KIKA_SMTP_* variables)."

    return True, "Verification email sent. Please check your inbox."


def resend_verification_email(email: str) -> Tuple[bool, str]:
    """
    Resend the verification link for an existing account.
    """
    _ensure_auth_tables()
    email = _normalize_email(email)
    if not email or not _EMAIL_RE.match(email):
        return False, "Enter a valid email address."

    with db_cursor() as cursor:
        cursor.execute(
            "SELECT full_name, email_verified FROM users WHERE email = ?",
            (email,),
        )
        row = cursor.fetchone()

    if not row:
        return False, "No account found with that email."

    if row["email_verified"]:
        return False, "This email is already verified."

    return send_verification_email(email, row["full_name"] or "there")


def mark_email_verified(email: str) -> bool:
    """
    Mark the specified email address as verified.
    """
    _ensure_auth_tables()
    email = _normalize_email(email)
    with db_cursor() as cursor:
        cursor.execute(
            "UPDATE users SET email_verified = 1 WHERE email = ?",
            (email,),
        )
        return cursor.rowcount > 0


def handle_verification_query() -> None:
    """
    Process `?verify=` query parameters and surface feedback to the user.
    """
    # Use st.query_params (new API)
    # st.query_params returns a dict-like object where values are strings (not lists)
    token = st.query_params.get("verify")
    if token:
        feedback: Dict[str, Any] = {"status": "error", "message": "Verification failed."}
        email_hint: Optional[str] = None
        try:
            email = confirm_email_verification_token(token)
        except VerificationTokenExpired as exc:
            email_hint = exc.email
            feedback = {
                "status": "expired",
                "message": "Your verification link has expired. Request a new one below.",
                "email": email_hint,
            }
        except VerificationTokenInvalid:
            feedback = {
                "status": "error",
                "message": "Verification link is invalid. Please request a new link below.",
                "email": None,
            }
        except RuntimeError as exc:
            feedback = {
                "status": "error",
                "message": f"Verification is currently unavailable: {exc}",
                "email": None,
            }
        else:
            if mark_email_verified(email):
                feedback = {"status": "success", "message": "Email verified! You can now sign in."}
            else:
                feedback = {"status": "info", "message": "Email already verified. You can sign in."}

        st.session_state[_VERIFICATION_FEEDBACK_KEY] = feedback
        # Remove the verify parameter from query params
        del st.query_params["verify"]

    feedback = st.session_state.pop(_VERIFICATION_FEEDBACK_KEY, None)
    if feedback:
        status = feedback.get("status", "info")
        message = feedback.get("message", "")
        display_fn = {
            "success": st.success,
            "info": st.info,
            "expired": st.warning,
            "error": st.error,
        }.get(status, st.info)
        display_fn(message)

        email_hint = feedback.get("email")
        if email_hint:
            if st.button("Resend verification email", key="resend_verification_postfeedback"):
                sent, resend_message = resend_verification_email(email_hint)
                if sent:
                    st.success(resend_message)
                else:
                    st.warning(resend_message)


def update_user_password(email: str, password: str) -> Tuple[bool, str]:
    """
    Update the password for an existing user identified by email.
    """
    _ensure_auth_tables()
    email = _normalize_email(email)

    if len(password) < 8 or len(password) > 128:
        return False, _PASSWORD_POLICY

    with db_cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if not row:
            return False, "User not found."

        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (_hash_password(password), row["id"]),
        )

    return True, "Password updated successfully."


def list_all_users() -> List[Dict[str, Any]]:
    """
    Return a list of user dictionaries for administration tasks.
    """
    _ensure_auth_tables()
    with db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, email, full_name, created_at, last_login, is_active, email_verified
            FROM users
            ORDER BY email
            """
        )
        rows = cursor.fetchall()
    return [
        {
            "id": row["id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "created_at": row["created_at"],
            "last_login": row["last_login"],
            "is_active": bool(row["is_active"]),
            "email_verified": bool(row["email_verified"]),
        }
        for row in rows
    ]


def set_user_active(email: str, is_active: bool) -> Tuple[bool, str]:
    """
    Activate or deactivate a user account.
    """
    _ensure_auth_tables()
    email = _normalize_email(email)

    with db_cursor() as cursor:
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if not row:
            return False, "User not found."

        cursor.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, row["id"]),
        )

    status = "activated" if is_active else "deactivated"
    return True, f"User {status}."


def render_account_sidebar(user: Dict[str, Any]) -> None:
    """
    Show account information and logout controls inside the sidebar.
    """
    with st.sidebar:
        st.markdown("### 👤 Account")
        st.markdown(f"**{user.get('full_name', 'Unknown User')}**")

        if user.get("email"):
            st.caption(user["email"])
        else:
            st.caption("Guest session")

        if user.get("is_guest"):
            st.info("Guest settings are temporary.", icon="ℹ️")

        if st.button("Sign out", key="auth_logout_button"):
            logout_user()
            st.success("Signed out.")
            st.rerun()
