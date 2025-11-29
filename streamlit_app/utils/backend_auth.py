"""
Authentication module using kika-backend API.

Provides the same interface as the old auth.py but connects to the 
kika-backend FastAPI server instead of using local SQLite.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from .api_client import (
    BackendConnectionError,
    check_backend_health,
    login_user as api_login,
    register_user as api_register,
    verify_email as api_verify,
    forgot_password as api_forgot_password,
    get_user_status,
    admin_list_users,
    admin_deactivate_user,
    admin_create_user,
    track_event,
)
from .user_settings import (
    USER_SETTINGS_KEY,
    USER_SETTINGS_OWNER_KEY,
    bootstrap_user_settings,
    get_default_settings,
)


AUTH_SESSION_KEY = "authenticated_user"
_VERIFICATION_FEEDBACK_KEY = "email_verification_feedback"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PASSWORD_POLICY = "Password must be at least 8 characters long."


def _normalize_email(email: str) -> str:
    """Normalize email to lowercase."""
    return email.strip().lower()


def _make_user_dict(email: str, verified: bool, is_active: bool, *, is_guest: bool = False) -> Dict[str, Any]:
    """Create a user dictionary for session state."""
    if is_guest:
        return {
            "email": None,
            "full_name": "Guest",
            "verified": False,
            "is_active": True,
            "is_guest": True,
        }
    return {
        "email": email,
        "full_name": email.split("@")[0].title(),  # Use email prefix as name
        "verified": verified,
        "is_active": is_active,
        "is_guest": False,
    }


def register_user(full_name: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user account via backend API.
    
    Note: full_name parameter kept for backward compatibility but not used
    since backend doesn't store it yet.
    """
    email = _normalize_email(email)
    
    if not email or not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    
    if len(password) < 8 or len(password) > 128:
        return False, _PASSWORD_POLICY
    
    success, message = api_register(email, password)
    
    # Track registration event if successful
    if success:
        track_event(email, "user_registered", {"source": "streamlit"})
    
    return success, message


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticate user via backend API.
    """
    email = _normalize_email(email)
    success, user_data, message = api_login(email, password)
    
    if success and user_data:
        # Convert backend user data to session format
        user = _make_user_dict(
            user_data["email"],
            user_data["verified"],
            user_data["is_active"]
        )
        
        # Track login event
        track_event(email, "user_logged_in", {"source": "streamlit"})
        
        return True, user, message
    
    return False, None, message


def logout_user() -> None:
    """Clear session state for the current user."""
    user = get_current_user()
    if user and user.get("email"):
        track_event(user["email"], "user_logged_out", {"source": "streamlit"})
    
    for key in (AUTH_SESSION_KEY, USER_SETTINGS_KEY, USER_SETTINGS_OWNER_KEY):
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear legacy keys
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
    """Return the authenticated user dictionary (or None)."""
    return st.session_state.get(AUTH_SESSION_KEY)


def require_user() -> Dict[str, Any]:
    """
    Guarantee an authenticated user is present, rendering the auth portal if required.
    """
    user = get_current_user()
    if user:
        bootstrap_user_settings(None, user.get("is_guest", False))
        return user
    
    _render_auth_portal()
    st.stop()


def _render_auth_portal() -> None:
    """Render the authentication portal with login/register/guest options."""
    st.title("🔐 Welcome to KIKA")
    st.markdown("Sign in to access your projects, or continue as a guest to explore with default settings.")
    
    # Check backend health
    if not check_backend_health():
        st.error(
            "⚠️ Unable to connect to backend server. "
            "Please ensure the kika-backend is running or continue as guest."
        )
    
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
                bootstrap_user_settings(None, False)
                st.success(message)
                st.rerun()
            else:
                if "not verified" in message.lower() or "verify" in message.lower():
                    st.warning(message)
                    st.session_state["auth_pending_email"] = _normalize_email(email)
                    st.info("Check your email for a verification link, or request a new one below.")
                else:
                    st.error(message)
        
        with st.expander("Forgot password?", expanded=False):
            with st.form("forgot_password_form"):
                reset_email = st.text_input("Email address", key="auth_reset_email")
                reset_submit = st.form_submit_button("Send reset link")
            
            if reset_submit:
                success, message = api_forgot_password(reset_email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
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
                    st.success("Account created! Check your email to verify your address.")
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
            guest_user = _make_user_dict("", False, True, is_guest=True)
            st.session_state[AUTH_SESSION_KEY] = guest_user
            st.session_state[USER_SETTINGS_KEY] = get_default_settings()
            st.session_state[USER_SETTINGS_OWNER_KEY] = "guest"
            bootstrap_user_settings(None, True)
            st.query_params.clear()
            st.success("Continuing as guest.")
            st.rerun()


def handle_verification_query() -> None:
    """
    Process `?verify=` query parameters and surface feedback to the user.
    """
    token = st.query_params.get("verify")
    if token:
        feedback: Dict[str, Any] = {"status": "error", "message": "Verification failed."}
        
        status, message = api_verify(token)
        
        if status == "success":
            feedback = {"status": "success", "message": message}
        elif status == "expired":
            feedback = {"status": "expired", "message": message}
        elif status == "info":
            feedback = {"status": "info", "message": message}
        else:
            feedback = {"status": "error", "message": message}
        
        st.session_state[_VERIFICATION_FEEDBACK_KEY] = feedback
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


def render_account_sidebar(user: Dict[str, Any]) -> None:
    """
    Show account information and logout controls inside the sidebar.
    """
    with st.sidebar:
        st.markdown("### 👤 Account")
        st.markdown(f"**{user.get('full_name', 'Unknown User')}**")
        
        if user.get("email"):
            st.caption(user["email"])
            if not user.get("verified", False):
                st.warning("⚠️ Email not verified", icon="⚠️")
        else:
            st.caption("Guest session")
        
        if user.get("is_guest"):
            st.info("Guest settings are temporary.", icon="ℹ️")
        
        if st.button("Sign out", key="auth_logout_button"):
            logout_user()
            st.success("Signed out.")
            st.rerun()


# ============================================================================
# Admin Functions (for backward compatibility with manage_users.py)
# ============================================================================

def list_all_users() -> List[Dict[str, Any]]:
    """
    Return a list of user dictionaries for administration tasks.
    """
    result = admin_list_users(limit=500)
    if not result:
        return []
    
    users = []
    for item in result.get("items", []):
        users.append({
            "id": item["id"],
            "email": item["email"],
            "full_name": item["email"].split("@")[0].title(),
            "created_at": item["created_at"],
            "last_login": item.get("last_login_at"),
            "is_active": item["is_active"],
            "email_verified": bool(item.get("verified_at")),
        })
    
    return users


def set_user_active(email: str, is_active: bool) -> Tuple[bool, str]:
    """
    Activate or deactivate a user account.
    """
    if is_active:
        # Backend doesn't have activate endpoint yet, would need to be added
        return False, "User activation not yet implemented in backend."
    else:
        return admin_deactivate_user(email)


def update_user_password(email: str, password: str) -> Tuple[bool, str]:
    """
    Update the password for an existing user.
    
    Note: This should be done via password reset flow in the backend.
    """
    return False, "Direct password updates not supported. Use password reset flow."


def send_verification_email(email: str, full_name: str, *, token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Send an email verification link to the user.
    
    Note: Backend handles this automatically on registration.
    """
    return True, "Verification emails are sent automatically by the backend on registration."


def resend_verification_email(email: str) -> Tuple[bool, str]:
    """
    Resend the verification link for an existing account.
    
    Note: Need to re-register to trigger new verification email.
    """
    email = _normalize_email(email)
    if not email or not _EMAIL_RE.match(email):
        return False, "Enter a valid email address."
    
    # Backend re-sends verification on re-registration
    success, message = api_register(email, "temporary_password_for_resend")
    if "already exists" in message.lower():
        return True, "If your email is registered, you'll receive a new verification link."
    
    return success, message


def mark_email_verified(email: str) -> bool:
    """
    Mark the specified email address as verified.
    
    Note: This is handled by the backend verify endpoint.
    """
    # This is done through the verify endpoint, not directly
    return False
