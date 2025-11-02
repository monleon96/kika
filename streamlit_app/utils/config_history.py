"""
Configuration History Management
Handles saving and restoring plot configurations for ACE and ENDF viewers.
"""

import streamlit as st
from datetime import datetime
import uuid
import copy


# Define which session state keys belong to each viewer
ACE_VIEWER_KEYS = [
    'uploaded_files', 'plot_selections', 'ace_files', 'ace_objects',
    'plot_title', 'x_label', 'y_label',
    'fig_width', 'fig_height', 'title_fontsize', 'label_fontsize',
    'legend_fontsize', 'show_legend', 'legend_loc',
    'log_x', 'log_y', 'show_grid', 'grid_alpha', 
    'show_minor_grid', 'minor_grid_alpha',
    'tick_fontsize_x', 'tick_fontsize_y', 'max_ticks_x', 'max_ticks_y',
    'rotate_x', 'rotate_y',
    'x_min', 'x_max', 'y_min', 'y_max'
]

ENDF_VIEWER_KEYS = [
    'endf_uploaded_files', 'endf_plot_selections', 'endf_files', 'endf_objects',
    'endf_plot_title', 'endf_x_label', 'endf_y_label',
    'endf_fig_width', 'endf_fig_height', 'endf_title_fontsize', 
    'endf_label_fontsize', 'endf_legend_fontsize', 'endf_show_legend', 
    'endf_legend_loc',
    'endf_log_x', 'endf_log_y', 'endf_show_grid', 'endf_grid_alpha',
    'endf_show_minor_grid', 'endf_minor_grid_alpha',
    'endf_tick_fontsize_x', 'endf_tick_fontsize_y', 
    'endf_max_ticks_x', 'endf_max_ticks_y',
    'endf_rotate_x', 'endf_rotate_y',
    'endf_x_min', 'endf_x_max', 'endf_y_min', 'endf_y_max'
]


def initialize_history():
    """Initialize the saved configurations list in session state."""
    if 'saved_configs' not in st.session_state:
        st.session_state.saved_configs = []


def save_configuration(viewer_type: str, config_name: str = None, config_id: str = None) -> dict:
    """
    Save the current viewer configuration.
    
    Parameters
    ----------
    viewer_type : str
        Either 'ace' or 'endf'
    config_name : str, optional
        Custom name for the configuration. If None, auto-generates one.
    config_id : str, optional
        If provided, updates existing configuration with this ID instead of creating new one.
    
    Returns
    -------
    dict
        The saved configuration object
    """
    initialize_history()
    
    # Determine which keys to save
    if viewer_type == 'ace':
        keys_to_save = ACE_VIEWER_KEYS
        default_name_prefix = "ACE"
    elif viewer_type == 'endf':
        keys_to_save = ENDF_VIEWER_KEYS
        default_name_prefix = "ENDF"
    else:
        raise ValueError(f"Unknown viewer type: {viewer_type}")
    
    # Create configuration snapshot
    config_snapshot = {}
    for key in keys_to_save:
        if key in st.session_state:
            # Deep copy to avoid reference issues
            config_snapshot[key] = copy.deepcopy(st.session_state[key])
    
    # Count uploaded files for metadata
    if viewer_type == 'ace':
        file_count = len(st.session_state.get('ace_objects', {}))
        series_count = len(st.session_state.get('plot_selections', []))
        file_names = list(st.session_state.get('ace_objects', {}).keys())
    else:
        file_count = len(st.session_state.get('endf_objects', {}))
        series_count = len(st.session_state.get('endf_plot_selections', []))
        file_names = list(st.session_state.get('endf_objects', {}).keys())
    
    # Generate auto name if not provided
    if config_name is None or config_name.strip() == "":
        # Create a descriptive auto-generated name
        timestamp_str = datetime.now().strftime("%b%d %H:%M")
        
        # Include first file name if available
        if file_names:
            first_file = file_names[0]
            # Truncate filename if too long
            if len(first_file) > 20:
                first_file = first_file[:17] + "..."
            config_name = f"{default_name_prefix} - {first_file} ({series_count} series)"
        else:
            config_name = f"{default_name_prefix} Config - {timestamp_str}"
    
    # Check if updating existing config
    if config_id:
        # Find and update the existing configuration
        for i, saved_config in enumerate(st.session_state.saved_configs):
            if saved_config['id'] == config_id:
                st.session_state.saved_configs[i] = {
                    'id': config_id,
                    'name': config_name,
                    'viewer': viewer_type,
                    'timestamp': datetime.now(),
                    'file_count': file_count,
                    'series_count': series_count,
                    'config': config_snapshot
                }
                return st.session_state.saved_configs[i]
    
    # Create new configuration object
    new_id = str(uuid.uuid4())
    config = {
        'id': new_id,
        'name': config_name,
        'viewer': viewer_type,
        'timestamp': datetime.now(),
        'file_count': file_count,
        'series_count': series_count,
        'config': config_snapshot
    }
    
    # Add to saved configurations
    st.session_state.saved_configs.append(config)
    
    # Track this as the current config
    if viewer_type == 'ace':
        st.session_state.ace_current_config_id = new_id
    else:
        st.session_state.endf_current_config_id = new_id
    
    return config


def restore_configuration(config_id: str):
    """
    Restore a saved configuration.
    
    Parameters
    ----------
    config_id : str
        The unique ID of the configuration to restore
    """
    initialize_history()
    
    # Find the configuration
    config = None
    for saved_config in st.session_state.saved_configs:
        if saved_config['id'] == config_id:
            config = saved_config
            break
    
    if config is None:
        st.error(f"Configuration with ID {config_id} not found")
        return
    
    # Restore all keys from the snapshot
    for key, value in config['config'].items():
        st.session_state[key] = copy.deepcopy(value)
    
    # Track this as the current config
    if config['viewer'] == 'ace':
        st.session_state.ace_current_config_id = config_id
    else:
        st.session_state.endf_current_config_id = config_id
    
    st.success(f"✓ Restored configuration: {config['name']}")


def delete_configuration(config_id: str):
    """
    Delete a saved configuration.
    
    Parameters
    ----------
    config_id : str
        The unique ID of the configuration to delete
    """
    initialize_history()
    
    # Remove the configuration
    st.session_state.saved_configs = [
        config for config in st.session_state.saved_configs 
        if config['id'] != config_id
    ]


def get_saved_configurations():
    """
    Get all saved configurations.
    
    Returns
    -------
    list
        List of saved configuration objects
    """
    initialize_history()
    return st.session_state.saved_configs


def clear_all_configurations():
    """Clear all saved configurations."""
    st.session_state.saved_configs = []


def get_config_by_name(config_name: str) -> dict:
    """
    Find a configuration by name.
    
    Parameters
    ----------
    config_name : str
        The name of the configuration to find
    
    Returns
    -------
    dict or None
        The configuration object if found, None otherwise
    """
    initialize_history()
    for config in st.session_state.saved_configs:
        if config['name'] == config_name:
            return config
    return None
