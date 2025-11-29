"""
Session state management utilities

Helpers for managing Streamlit session state
"""

import streamlit as st
from typing import Any, Dict, Optional


def init_session_state(key: str, default_value: Any) -> None:
    """
    Initialize a session state variable if it doesn't exist
    
    Args:
        key: Session state key
        default_value: Default value to set if key doesn't exist
    """
    if key not in st.session_state:
        st.session_state[key] = default_value


def get_session_state(key: str, default: Any = None) -> Any:
    """
    Get a value from session state with a default fallback
    
    Args:
        key: Session state key
        default: Default value if key doesn't exist
        
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_state(key: str, value: Any) -> None:
    """
    Set a value in session state
    
    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def clear_session_state(*keys: str) -> None:
    """
    Clear specific keys from session state
    
    Args:
        keys: Session state keys to clear
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def reset_session_state() -> None:
    """
    Reset all session state (use with caution!)
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]


class SessionStateManager:
    """
    Context manager for session state operations
    """
    
    def __init__(self, namespace: str):
        """
        Initialize with a namespace to avoid key collisions
        
        Args:
            namespace: Prefix for all keys
        """
        self.namespace = namespace
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}_{key}"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get namespaced value"""
        return get_session_state(self._make_key(key), default)
    
    def set(self, key: str, value: Any) -> None:
        """Set namespaced value"""
        set_session_state(self._make_key(key), value)
    
    def init(self, key: str, default_value: Any) -> None:
        """Initialize namespaced value"""
        init_session_state(self._make_key(key), default_value)
    
    def clear(self, *keys: str) -> None:
        """Clear namespaced keys"""
        namespaced_keys = [self._make_key(k) for k in keys]
        clear_session_state(*namespaced_keys)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all values in this namespace"""
        prefix = f"{self.namespace}_"
        return {
            k.replace(prefix, ""): v
            for k, v in st.session_state.items()
            if k.startswith(prefix)
        }
