"""
API Client for KIKA Backend

Provides functions to interact with the kika-backend FastAPI server.
Handles authentication, user management, and other backend operations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests
from requests.exceptions import ConnectionError, RequestException, Timeout


def get_backend_url() -> str:
    """
    Get the backend API URL from environment or use default.
    
    Returns:
        Backend URL (default: http://localhost:8000)
    """
    return os.getenv("KIKA_BACKEND_URL", "http://localhost:8000").rstrip("/")


def get_admin_key() -> Optional[str]:
    """
    Get the admin API key from environment.
    
    Returns:
        Admin API key or None
    """
    return os.getenv("KIKA_ADMIN_KEY")


class BackendError(Exception):
    """Base exception for backend API errors."""
    pass


class BackendConnectionError(BackendError):
    """Raised when unable to connect to backend."""
    pass


class BackendAuthError(BackendError):
    """Raised when authentication fails."""
    pass


class BackendValidationError(BackendError):
    """Raised when request validation fails."""
    pass


def _handle_response(response: requests.Response) -> Dict[str, Any]:
    """
    Process API response and raise appropriate exceptions.
    
    Args:
        response: requests Response object
        
    Returns:
        Response JSON data
        
    Raises:
        BackendAuthError: For 401 Unauthorized
        BackendValidationError: For 400 Bad Request
        BackendError: For other errors
    """
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise BackendAuthError(response.json().get("detail", "Unauthorized"))
        elif response.status_code == 400:
            raise BackendValidationError(response.json().get("detail", "Bad request"))
        elif response.status_code == 404:
            raise BackendError(response.json().get("detail", "Not found"))
        elif response.status_code == 429:
            raise BackendError("Too many requests. Please try again later.")
        else:
            raise BackendError(f"API error: {response.json().get('detail', str(e))}")


# ============================================================================
# Authentication Endpoints
# ============================================================================

def register_user(email: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user account.
    
    Args:
        email: User's email address
        password: User's password (min 8 characters)
        
    Returns:
        (success: bool, message: str)
    """
    try:
        url = f"{get_backend_url()}/register"
        payload = {"email": email, "password": password}
        response = requests.post(url, json=payload, timeout=30)  # Aumentar timeout para Render
        _handle_response(response)
        return True, "Account created! Check your email to verify your address."
    except BackendConnectionError:
        return False, "Unable to connect to backend. Please try again later."
    except BackendValidationError as e:
        return False, str(e)
    except BackendError as e:
        return False, str(e)
    except (ConnectionError, Timeout):
        return False, "Connection timeout. Please check your internet connection."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def login_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticate a user.
    
    Args:
        email: User's email address
        password: User's password
        
    Returns:
        (success: bool, user_data: dict or None, message: str)
    """
    try:
        url = f"{get_backend_url()}/login"
        payload = {"email": email, "password": password}
        response = requests.post(url, json=payload, timeout=30)  # Aumentar timeout para Render
        _handle_response(response)
        
        # Login successful - fetch user details
        user_data = get_user_status(email)
        if user_data:
            return True, user_data, "Signed in successfully."
        else:
            return False, None, "Login succeeded but couldn't fetch user data."
            
    except BackendAuthError as e:
        return False, None, "Invalid email or password."
    except BackendConnectionError:
        return False, None, "Unable to connect to backend."
    except BackendError as e:
        return False, None, str(e)
    except (ConnectionError, Timeout):
        return False, None, "Connection timeout. Please try again."
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"


def verify_email(token: str) -> Tuple[str, str]:
    """
    Verify email address with token.
    
    Args:
        token: Email verification token
        
    Returns:
        (status: str, message: str)
        status can be: 'success', 'error', 'expired', 'info'
    """
    try:
        url = f"{get_backend_url()}/verify"
        params = {"token": token}
        response = requests.get(url, params=params, timeout=30)
        
        # Backend returns HTML, but we check status code
        if response.status_code == 200:
            return "success", "Email verified! You can now sign in."
        elif response.status_code == 404:
            return "error", "User not found."
        elif response.status_code == 400:
            error_detail = response.json().get("detail", "")
            if "expired" in error_detail.lower():
                return "expired", "Verification link has expired. Request a new one."
            else:
                return "error", "Verification link is invalid."
        else:
            return "error", "Verification failed. Please try again."
            
    except (ConnectionError, Timeout):
        return "error", "Connection timeout. Please try again."
    except Exception as e:
        return "error", f"Verification failed: {str(e)}"


def forgot_password(email: str) -> Tuple[bool, str]:
    """
    Request password reset email.
    
    Args:
        email: User's email address
        
    Returns:
        (success: bool, message: str)
    """
    try:
        url = f"{get_backend_url()}/password/forgot"
        payload = {"email": email}
        response = requests.post(url, json=payload, timeout=30)
        _handle_response(response)
        return True, "If an account exists with that email, you'll receive reset instructions."
    except BackendError as e:
        return False, str(e)
    except (ConnectionError, Timeout):
        return False, "Connection timeout. Please try again."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def reset_password(token: str, new_password: str) -> Tuple[bool, str]:
    """
    Reset password with token.
    
    Args:
        token: Password reset token
        new_password: New password
        
    Returns:
        (success: bool, message: str)
    """
    try:
        url = f"{get_backend_url()}/password/reset"
        payload = {"token": token, "new_password": new_password}
        response = requests.post(url, json=payload, timeout=30)
        _handle_response(response)
        return True, "Password reset successful! You can now sign in."
    except BackendValidationError as e:
        return False, str(e)
    except BackendError as e:
        return False, str(e)
    except (ConnectionError, Timeout):
        return False, "Connection timeout. Please try again."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# ============================================================================
# User Management Endpoints
# ============================================================================

def get_user_status(email: str) -> Optional[Dict[str, Any]]:
    """
    Get user status information.
    
    Args:
        email: User's email address
        
    Returns:
        User data dict or None if not found
    """
    try:
        url = f"{get_backend_url()}/users/{email}"
        response = requests.get(url, timeout=30)
        data = _handle_response(response)
        return {
            "email": data["email"],
            "verified": data["verified"],
            "is_active": data["is_active"],
        }
    except BackendError:
        return None
    except (ConnectionError, Timeout):
        return None
    except Exception:
        return None


# ============================================================================
# Admin Endpoints (require X-Admin-Key header)
# ============================================================================

def admin_create_user(email: str, password: Optional[str] = None, 
                     verified: bool = False, is_active: bool = True) -> Tuple[bool, str]:
    """
    Admin: Create a new user.
    
    Args:
        email: User's email
        password: User's password (optional)
        verified: Whether email is verified
        is_active: Whether user is active
        
    Returns:
        (success: bool, message: str)
    """
    admin_key = get_admin_key()
    if not admin_key:
        return False, "Admin key not configured."
    
    try:
        url = f"{get_backend_url()}/admin/users/create"
        headers = {"X-Admin-Key": admin_key}
        payload = {
            "email": email,
            "password": password,
            "verified": verified,
            "is_active": is_active,
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        _handle_response(response)
        return True, f"User {email} created successfully."
    except BackendAuthError:
        return False, "Invalid admin key."
    except BackendError as e:
        return False, str(e)
    except (ConnectionError, Timeout):
        return False, "Connection timeout."
    except Exception as e:
        return False, f"Error: {str(e)}"


def admin_deactivate_user(email: str) -> Tuple[bool, str]:
    """
    Admin: Deactivate a user account.
    
    Args:
        email: User's email
        
    Returns:
        (success: bool, message: str)
    """
    admin_key = get_admin_key()
    if not admin_key:
        return False, "Admin key not configured."
    
    try:
        url = f"{get_backend_url()}/admin/users/deactivate"
        headers = {"X-Admin-Key": admin_key}
        payload = {"email": email}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        _handle_response(response)
        return True, f"User {email} deactivated."
    except BackendAuthError:
        return False, "Invalid admin key."
    except BackendError as e:
        return False, str(e)
    except (ConnectionError, Timeout):
        return False, "Connection timeout."
    except Exception as e:
        return False, f"Error: {str(e)}"


def admin_list_users(limit: int = 100, offset: int = 0) -> Optional[Dict[str, Any]]:
    """
    Admin: List all users.
    
    Args:
        limit: Maximum number of users to return
        offset: Number of users to skip
        
    Returns:
        Dict with 'items' (list of users) and 'count' (total count), or None
    """
    admin_key = get_admin_key()
    if not admin_key:
        return None
    
    try:
        url = f"{get_backend_url()}/admin/users/list"
        headers = {"X-Admin-Key": admin_key}
        params = {"limit": limit, "offset": offset}
        response = requests.get(url, headers=headers, params=params, timeout=30)
        return _handle_response(response)
    except Exception:
        return None


# ============================================================================
# Metrics/Analytics Endpoints
# ============================================================================

def track_event(email: str, event_name: str, props: Optional[Dict[str, Any]] = None) -> bool:
    """
    Track an analytics event.
    
    Args:
        email: User's email
        event_name: Event name
        props: Optional event properties
        
    Returns:
        Success boolean
    """
    try:
        url = f"{get_backend_url()}/metrics/event"
        payload = {
            "email": email,
            "event_name": event_name,
            "props": props or {},
        }
        response = requests.post(url, json=payload, timeout=30)
        _handle_response(response)
        return True
    except Exception:
        # Don't fail if analytics tracking fails
        return False


# ============================================================================
# Health Check
# ============================================================================

def check_backend_health() -> bool:
    """
    Check if backend is reachable and healthy.
    
    Returns:
        True if backend is healthy, False otherwise
    """
    try:
        url = f"{get_backend_url()}/healthz"
        response = requests.get(url, timeout=30)  # Aumentar timeout para Render free tier
        return response.status_code == 200
    except Exception:
        return False
