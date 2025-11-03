"""
Utility modules for KIKA Streamlit app
"""

from .session_state import (
    init_session_state,
    get_session_state,
    set_session_state,
    clear_session_state,
    reset_session_state,
    SessionStateManager,
)

from .file_utils import (
    save_uploaded_file,
    cleanup_temp_files,
    get_file_info,
    format_file_size,
    validate_ace_file,
    FileManager,
)

from .auth import (
    require_user,
    get_current_user,
    render_account_sidebar,
    logout_user,
    update_user_password,
    list_all_users,
    set_user_active,
    send_verification_email,
    resend_verification_email,
    mark_email_verified,
    handle_verification_query,
)

from .user_settings import (
    get_current_settings,
    update_setting,
    reset_settings_to_defaults,
    collect_settings_snapshot,
    save_user_settings,
    bootstrap_user_settings,
)

__all__ = [
    # Session state
    'init_session_state',
    'get_session_state',
    'set_session_state',
    'clear_session_state',
    'reset_session_state',
    'SessionStateManager',
    # File utils
    'save_uploaded_file',
    'cleanup_temp_files',
    'get_file_info',
    'format_file_size',
    'validate_ace_file',
    'FileManager',
    # Auth
    'require_user',
    'get_current_user',
    'render_account_sidebar',
    'logout_user',
    'update_user_password',
    'list_all_users',
    'set_user_active',
    'send_verification_email',
    'resend_verification_email',
    'mark_email_verified',
    'handle_verification_query',
    # User settings
    'get_current_settings',
    'update_setting',
    'reset_settings_to_defaults',
    'collect_settings_snapshot',
    'save_user_settings',
    'bootstrap_user_settings',
]
