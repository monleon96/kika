"""
Sidebar component for displaying saved plots.
"""

import streamlit as st
from utils.config_history import (
    get_saved_configurations,
    restore_configuration,
    delete_configuration,
    clear_all_configurations
)


def render_saved_configs_sidebar():
    """
    Render the saved plots panel in the sidebar.
    Should be called in the sidebar of each viewer page.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💾 Saved Plots")
    
    saved_configs = get_saved_configurations()
    
    if not saved_configs:
        st.sidebar.info("No saved plots yet.\n\nUse the **Save Plot** button in any viewer to save your current plot setup.")
        return
    
    # Show count and clear all button
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.markdown(f"**{len(saved_configs)} saved**")
    with col2:
        if st.button("🗑️", key="clear_all_configs", help="Clear all saved plots"):
            clear_all_configurations()
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Display each saved configuration (oldest first)
    for config in saved_configs:
        render_config_card(config)


def render_config_card(config: dict):
    """
    Render a single plot card.
    
    Parameters
    ----------
    config : dict
        Plot configuration object containing metadata and settings
    """
    # Determine icon and color based on viewer type
    if config['viewer'] == 'ace':
        icon = "📊"
        color = "#1f77b4"
        viewer_label = "ACE Viewer"
    else:
        icon = "📈"
        color = "#ff7f0e"
        viewer_label = "ENDF Viewer"
    
    # Create expander with config name
    with st.sidebar.expander(f"{icon} {config['name']}", expanded=False):
        # Metadata
        st.markdown(f"**Viewer:** {viewer_label}")
        st.markdown(f"**Files:** {config['file_count']} | **Series:** {config['series_count']}")
        
        # Format timestamp
        timestamp = config['timestamp']
        time_str = timestamp.strftime("%b %d, %H:%M:%S")
        st.markdown(f"**Saved:** {time_str}")
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("↩️ Restore", key=f"restore_{config['id']}", 
                        help="Restore this plot", width="stretch"):
                restore_configuration(config['id'])
                
                # Navigate to the correct viewer if needed
                current_page = st.session_state.get('current_page', None)
                target_page = "pages/1_📊_ACE_Viewer.py" if config['viewer'] == 'ace' else "pages/2_📈_ENDF_Viewer.py"
                
                # Check if we need to switch pages
                if current_page and current_page != target_page:
                    st.switch_page(target_page)
                else:
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{config['id']}", 
                        help="Delete this plot", width="stretch"):
                delete_configuration(config['id'])
                st.rerun()
