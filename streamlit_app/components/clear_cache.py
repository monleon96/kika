"""
Shared sidebar component for clearing Streamlit and module caches.
"""

import streamlit as st
import sys


def render_clear_cache_button(key_prefix: str = "clear_cache"):
    """Render a standardized Clear Cache button in the sidebar.

    Parameters
    ----------
    key_prefix : str
        Prefix used to build unique Streamlit widget keys so the button can
        be placed on multiple pages without key collisions.
    """
    st.sidebar.markdown("---")
    button_key = f"{key_prefix}_{st.session_state.get('current_page','') or ''}"
    if st.sidebar.button("🔄 Clear Cache", key=button_key, help="Clear cache and reload modules", width="stretch"):
        # Clear Streamlit cache
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        # Force reimport of modules by clearing Python's module cache for kika
        modules_to_reload = [k for k in list(sys.modules.keys()) if k.startswith('kika')]
        for module in modules_to_reload:
            try:
                del sys.modules[module]
            except Exception:
                pass
        st.success("✓ Cache cleared! Refresh to reload.")
        # Try to trigger a rerun in a way that's compatible with multiple
        # Streamlit versions. `experimental_rerun` exists in some versions,
        # newer versions may expose `rerun`, otherwise fall back to stopping
        # the script which effectively ends execution until the user refreshes.
        try:
            if hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()
            elif hasattr(st, 'rerun'):
                st.rerun()
            else:
                # Best-effort fallback
                st.stop()
        except Exception:
            try:
                if hasattr(st, 'rerun'):
                    st.rerun()
                else:
                    st.stop()
            except Exception:
                # If all else fails, do nothing; the success message is shown
                pass
