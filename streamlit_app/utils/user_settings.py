"""
User settings management for the Streamlit UI.

Handles default values and synchronization with the Streamlit session state
so that other parts of the app can continue to read from `st.session_state.njoy_exe_path`
and similar keys.

Note: Settings are currently stored in session state only. For persistent storage,
a backend endpoint should be added to kika-backend in the future.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, MutableMapping, Sequence, Tuple

import streamlit as st


USER_SETTINGS_KEY = "user_settings"
USER_SETTINGS_OWNER_KEY = "user_settings_owner"


DEFAULT_SETTINGS: Dict[str, Any] = {
    "appearance": {
        "theme": "Auto",
        "layout": "Wide",
        "sidebar_state": "Expanded",
        "font_size": 14,
        "code_font": "Monospace",
        "high_contrast": False,
        "reduce_animations": False,
        "screen_reader": False,
    },
    "plot_defaults": {
        "width": 10,
        "height": 6,
        "dpi": 150,
        "grid": True,
        "legend": True,
        "legend_loc": "best",
        "linewidth": 2.0,
        "markersize": 6,
        "color_palette": "Default",
        "colors": [
            "#667eea",
            "#ff6b6b",
            "#4ecdc4",
            "#ffe66d",
            "#95e1d3",
        ],
    },
    "export": {
        "image_format": "PNG",
        "dpi": 300,
        "transparent_background": False,
        "filename_template": "kika_{datatype}_{timestamp}",
        "data_format": "CSV",
    },
    "njoy": {
        "exe_path": "/usr/local/bin/njoy",
        "version": "NJOY 2016.78",
        "output_dir": "./njoy_output",
        "create_xsdir": True,
        "auto_version": True,
    },
    "profile": {
        "auto_save": False,
        "email_notifications": False,
        "share_analytics": False,
    },
}


def _deep_merge(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries without mutating inputs.
    """
    merged: Dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = deepcopy(value)
        else:
            merged[key] = deepcopy(value)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_default_settings() -> Dict[str, Any]:
    """
    Return a fresh copy of the default settings dictionary.
    """
    return deepcopy(DEFAULT_SETTINGS)


def normalize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a settings dictionary contains all expected keys.
    """
    if not settings:
        return get_default_settings()
    return _deep_merge(get_default_settings(), settings)


def load_user_settings(user_id: int | None) -> Dict[str, Any]:
    """
    Load settings for the given user id (or defaults for guests).
    
    Note: Currently returns defaults for all users. Backend persistence
    should be implemented in the future.
    """
    # TODO: Load from backend API when user settings endpoints are available
    return get_default_settings()


def save_user_settings(user_id: int | None, settings: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Persist settings for a specific user. Guests cannot persist settings.
    
    Note: Currently only saves to session state. Backend persistence
    should be implemented in the future.
    """
    if user_id is None:
        return False, "Guest sessions use temporary settings and cannot be saved."

    normalized = normalize_settings(settings)
    
    # TODO: Save to backend API when user settings endpoints are available
    # For now, just keep in session state
    st.session_state[USER_SETTINGS_KEY] = normalized
    
    return True, "Settings saved to session (temporary until backend persistence is added)."


def _apply_settings_to_session_state(settings: Dict[str, Any]) -> None:
    """
    Mirror frequently used values to legacy session-state keys so existing
    pages do not need to be rewritten all at once.
    """
    njoy_settings = settings.get("njoy", {})
    st.session_state.njoy_exe_path = njoy_settings.get("exe_path", "/usr/local/bin/njoy")
    st.session_state.njoy_version = njoy_settings.get("version", "NJOY 2016.78")
    st.session_state.njoy_output_dir = njoy_settings.get("output_dir", "./njoy_output")
    st.session_state.njoy_create_xsdir = njoy_settings.get("create_xsdir", True)
    st.session_state.njoy_auto_version = njoy_settings.get("auto_version", True)


def bootstrap_user_settings(user_id: int | None, is_guest: bool) -> Dict[str, Any]:
    """
    Ensure `st.session_state` has an up-to-date settings dict for the active user.
    """
    owner_token = "guest" if is_guest or user_id is None else f"user-{user_id}"

    cached_owner = st.session_state.get(USER_SETTINGS_OWNER_KEY)
    cached_settings = st.session_state.get(USER_SETTINGS_KEY)
    if cached_settings is not None and cached_owner == owner_token:
        _apply_settings_to_session_state(cached_settings)
        return cached_settings

    settings = load_user_settings(None if is_guest else user_id)
    st.session_state[USER_SETTINGS_KEY] = settings
    st.session_state[USER_SETTINGS_OWNER_KEY] = owner_token
    _apply_settings_to_session_state(settings)
    return settings


def get_current_settings() -> Dict[str, Any]:
    """
    Return the settings dict currently loaded in session state.
    """
    settings = st.session_state.get(USER_SETTINGS_KEY)
    if settings is None:
        settings = get_default_settings()
        st.session_state[USER_SETTINGS_KEY] = settings
        st.session_state[USER_SETTINGS_OWNER_KEY] = "guest"
    return settings


def update_setting(path: Sequence[str], value: Any) -> None:
    """
    Update a nested setting using a key path.
    """
    if not path:
        raise ValueError("Setting path cannot be empty")

    settings = get_current_settings()
    node = settings
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value

    st.session_state[USER_SETTINGS_KEY] = settings

    # Sync derived session-state keys when NJOY values change.
    if path[0] == "njoy":
        _apply_settings_to_session_state(settings)


def reset_settings_to_defaults(user_id: int | None, is_guest: bool) -> Dict[str, Any]:
    """
    Replace the in-session settings with defaults.
    """
    defaults = get_default_settings()
    st.session_state[USER_SETTINGS_KEY] = defaults
    owner_token = "guest" if is_guest or user_id is None else f"user-{user_id}"
    st.session_state[USER_SETTINGS_OWNER_KEY] = owner_token
    _apply_settings_to_session_state(defaults)
    return defaults


def collect_settings_snapshot() -> Dict[str, Any]:
    """
    Return a deep copy of the current settings dictionary.
    """
    return deepcopy(get_current_settings())
