"""
Settings Page

Configure application preferences and user settings
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence

import streamlit as st

# Add repository root so user settings/auth utilities are importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.backend_auth import handle_verification_query, render_account_sidebar, require_user
from utils.user_settings import (
    DEFAULT_SETTINGS,
    collect_settings_snapshot,
    get_current_settings,
    reset_settings_to_defaults,
    save_user_settings,
    update_setting,
)


def _safe_index(options: Sequence, value, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


# Page config
st.set_page_config(page_title="Settings - KIKA", page_icon="⚙️", layout="wide")

handle_verification_query()
current_user = require_user()
render_account_sidebar(current_user)

settings = get_current_settings()
appearance_settings = settings.get("appearance", {})
plot_settings = settings.get("plot_defaults", {})
export_settings = settings.get("export", {})
njoy_settings = settings.get("njoy", {})
profile_settings = settings.get("profile", {})

# Header
st.title("⚙️ Settings")
st.markdown("Configure your KIKA experience")
st.markdown("---")

# Settings categories
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🎨 Appearance", "📊 Plot Defaults", "💾 Export", "🔧 NJOY", "👤 Profile"]
)

with tab1:
    st.header("Appearance Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Theme")
        theme_options = ["Light", "Dark", "Auto"]
        theme = st.selectbox(
            "Color theme",
            theme_options,
            index=_safe_index(theme_options, appearance_settings.get("theme", "Auto"), 2),
            help="Choose your preferred color scheme",
        )
        if theme != appearance_settings.get("theme"):
            update_setting(("appearance", "theme"), theme)

        st.info("ℹ️ Theme changes will be applied in future versions")

        st.subheader("Layout")
        layout_options = ["Wide", "Centered"]
        default_layout = st.selectbox(
            "Default layout",
            layout_options,
            index=_safe_index(layout_options, appearance_settings.get("layout", "Wide")),
            help="Default page layout",
        )
        if default_layout != appearance_settings.get("layout"):
            update_setting(("appearance", "layout"), default_layout)

        sidebar_options = ["Expanded", "Collapsed", "Auto"]
        sidebar_state = st.selectbox(
            "Sidebar default state",
            sidebar_options,
            index=_safe_index(
                sidebar_options,
                appearance_settings.get("sidebar_state", "Expanded"),
                0,
            ),
            help="How the sidebar appears on page load",
        )
        if sidebar_state != appearance_settings.get("sidebar_state"):
            update_setting(("appearance", "sidebar_state"), sidebar_state)

    with col2:
        st.subheader("Font & Size")
        font_size = st.slider(
            "Base font size",
            12,
            20,
            value=int(appearance_settings.get("font_size", 14)),
        )
        if font_size != appearance_settings.get("font_size"):
            update_setting(("appearance", "font_size"), font_size)

        font_options = ["Monospace", "Consolas", "Monaco"]
        code_font = st.selectbox(
            "Code font",
            font_options,
            index=_safe_index(font_options, appearance_settings.get("code_font", "Monospace")),
        )
        if code_font != appearance_settings.get("code_font"):
            update_setting(("appearance", "code_font"), code_font)

        st.subheader("Accessibility")
        high_contrast = st.checkbox(
            "High contrast mode", value=bool(appearance_settings.get("high_contrast", False))
        )
        if high_contrast != appearance_settings.get("high_contrast"):
            update_setting(("appearance", "high_contrast"), high_contrast)

        reduce_animations = st.checkbox(
            "Reduce animations", value=bool(appearance_settings.get("reduce_animations", False))
        )
        if reduce_animations != appearance_settings.get("reduce_animations"):
            update_setting(("appearance", "reduce_animations"), reduce_animations)

        screen_reader = st.checkbox(
            "Screen reader optimizations",
            value=bool(appearance_settings.get("screen_reader", False)),
        )
        if screen_reader != appearance_settings.get("screen_reader"):
            update_setting(("appearance", "screen_reader"), screen_reader)

with tab2:
    st.header("Plot Default Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Figure Size")
        default_width = st.slider(
            "Default width (inches)",
            6,
            16,
            value=int(plot_settings.get("width", DEFAULT_SETTINGS["plot_defaults"]["width"])),
        )
        if default_width != plot_settings.get("width"):
            update_setting(("plot_defaults", "width"), default_width)

        default_height = st.slider(
            "Default height (inches)",
            4,
            12,
            value=int(plot_settings.get("height", DEFAULT_SETTINGS["plot_defaults"]["height"])),
        )
        if default_height != plot_settings.get("height"):
            update_setting(("plot_defaults", "height"), default_height)

        dpi_options = [100, 150, 200, 300]
        default_dpi = st.selectbox(
            "Default DPI",
            dpi_options,
            index=_safe_index(dpi_options, plot_settings.get("dpi", 150), 1),
        )
        if default_dpi != plot_settings.get("dpi"):
            update_setting(("plot_defaults", "dpi"), default_dpi)

        st.subheader("Axes")
        default_grid = st.checkbox(
            "Show grid by default", value=bool(plot_settings.get("grid", True))
        )
        if default_grid != plot_settings.get("grid"):
            update_setting(("plot_defaults", "grid"), default_grid)

        default_legend = st.checkbox(
            "Show legend by default", value=bool(plot_settings.get("legend", True))
        )
        if default_legend != plot_settings.get("legend"):
            update_setting(("plot_defaults", "legend"), default_legend)

        legend_options = ["best", "upper right", "upper left", "lower right", "lower left"]
        default_legend_loc = st.selectbox(
            "Legend location",
            legend_options,
            index=_safe_index(
                legend_options, plot_settings.get("legend_loc", "best"), 0
            ),
        )
        if default_legend_loc != plot_settings.get("legend_loc"):
            update_setting(("plot_defaults", "legend_loc"), default_legend_loc)

    with col2:
        st.subheader("Line Styles")
        default_linewidth = st.slider(
            "Line width",
            0.5,
            5.0,
            value=float(plot_settings.get("linewidth", 2.0)),
            step=0.5,
        )
        if default_linewidth != plot_settings.get("linewidth"):
            update_setting(("plot_defaults", "linewidth"), default_linewidth)

        default_markersize = st.slider(
            "Marker size",
            2,
            12,
            value=int(plot_settings.get("markersize", 6)),
        )
        if default_markersize != plot_settings.get("markersize"):
            update_setting(("plot_defaults", "markersize"), default_markersize)

        st.subheader("Colors")
        palette_options = [
            "Default",
            "Colorblind-friendly",
            "Grayscale",
            "Vibrant",
            "Pastel",
        ]
        color_palette = st.selectbox(
            "Default color palette",
            palette_options,
            index=_safe_index(
                palette_options, plot_settings.get("color_palette", "Default"), 0
            ),
        )
        if color_palette != plot_settings.get("color_palette"):
            update_setting(("plot_defaults", "color_palette"), color_palette)

        base_colors = DEFAULT_SETTINGS["plot_defaults"]["colors"]
        current_colors = list(plot_settings.get("colors", base_colors))
        while len(current_colors) < len(base_colors):
            current_colors.append(base_colors[len(current_colors)])

        st.markdown("**Custom colors:**")
        color_columns = st.columns(len(base_colors))
        selected_colors = []
        for i, column in enumerate(color_columns):
            with column:
                color_value = st.color_picker(
                    f"Color {i + 1}",
                    current_colors[i],
                )
                selected_colors.append(color_value)

        if selected_colors != current_colors:
            update_setting(("plot_defaults", "colors"), selected_colors)

with tab3:
    st.header("Export Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Export Format")
        image_formats = ["PNG", "PDF", "SVG", "JPG"]
        default_format = st.selectbox(
            "Image format",
            image_formats,
            index=_safe_index(image_formats, export_settings.get("image_format", "PNG")),
            help="Default format for downloading plots",
        )
        if default_format != export_settings.get("image_format"):
            update_setting(("export", "image_format"), default_format)

        export_dpi_options = [100, 150, 200, 300, 600]
        export_dpi = st.selectbox(
            "Export DPI",
            export_dpi_options,
            index=_safe_index(export_dpi_options, export_settings.get("dpi", 300), 3),
            help="Higher DPI = better quality but larger file size",
        )
        if export_dpi != export_settings.get("dpi"):
            update_setting(("export", "dpi"), export_dpi)

        transparent_bg = st.checkbox(
            "Transparent background (PNG/SVG)",
            value=bool(export_settings.get("transparent_background", False)),
        )
        if transparent_bg != export_settings.get("transparent_background"):
            update_setting(("export", "transparent_background"), transparent_bg)

    with col2:
        st.subheader("File Naming")
        filename_template = st.text_input(
            "Filename template",
            value=export_settings.get(
                "filename_template",
                DEFAULT_SETTINGS["export"]["filename_template"],
            ),
            help="Available variables: {datatype}, {timestamp}, {mt}, {energy}",
        )
        if filename_template != export_settings.get("filename_template"):
            update_setting(("export", "filename_template"), filename_template)

        st.markdown("**Example filename:**")
        st.code("kika_cross_section_2024-11-01_143052.png")

        st.subheader("Data Export")
        data_formats = ["CSV", "JSON", "HDF5", "Parquet"]
        data_export_format = st.selectbox(
            "Data format",
            data_formats,
            index=_safe_index(
                data_formats, export_settings.get("data_format", "CSV"), 0
            ),
            help="Format for exporting numerical data",
        )
        if data_export_format != export_settings.get("data_format"):
            update_setting(("export", "data_format"), data_export_format)

with tab4:
    st.header("NJOY Configuration")
    st.markdown("Configure NJOY nuclear data processing tools")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NJOY Executable")
        njoy_exe = st.text_input(
            "Path to NJOY executable",
            value=njoy_settings.get("exe_path", DEFAULT_SETTINGS["njoy"]["exe_path"]),
            help="Full path to your NJOY executable (e.g., /home/user/NJOY2016/build/njoy)",
        )
        if njoy_exe != njoy_settings.get("exe_path"):
            update_setting(("njoy", "exe_path"), njoy_exe)

        if njoy_exe and os.path.exists(njoy_exe):
            st.success(f"✓ Executable found: {njoy_exe}")
        elif njoy_exe:
            st.warning(f"⚠️ File not found: {njoy_exe}")

        st.markdown("---")

        njoy_version = st.text_input(
            "NJOY version string",
            value=njoy_settings.get("version", DEFAULT_SETTINGS["njoy"]["version"]),
            help="Version string for metadata (e.g., 'NJOY 2016.78')",
        )
        if njoy_version != njoy_settings.get("version"):
            update_setting(("njoy", "version"), njoy_version)

    with col2:
        st.subheader("Output Settings")
        output_dir = st.text_input(
            "Default output directory",
            value=njoy_settings.get("output_dir", DEFAULT_SETTINGS["njoy"]["output_dir"]),
            help="Base directory for NJOY output files",
        )
        if output_dir != njoy_settings.get("output_dir"):
            update_setting(("njoy", "output_dir"), output_dir)

        st.markdown("---")

        st.subheader("Processing Options")
        create_xsdir = st.checkbox(
            "Create XSDIR files by default",
            value=bool(njoy_settings.get("create_xsdir", True)),
            help="Automatically generate XSDIR files for MCNP",
        )
        if create_xsdir != njoy_settings.get("create_xsdir"):
            update_setting(("njoy", "create_xsdir"), create_xsdir)

        auto_version = st.checkbox(
            "Automatic file versioning",
            value=bool(njoy_settings.get("auto_version", True)),
            help="Automatically version files to prevent overwrites",
        )
        if auto_version != njoy_settings.get("auto_version"):
            update_setting(("njoy", "auto_version"), auto_version)

    st.markdown("---")

    st.subheader("🧪 Test NJOY Installation")
    col_test1, col_test2 = st.columns([2, 1])

    with col_test1:
        st.markdown("Verify that NJOY is properly installed and accessible")

    with col_test2:
        if st.button("Run Test"):
            if not njoy_exe or not os.path.exists(njoy_exe):
                st.error("❌ Please set a valid NJOY executable path first")
            else:
                with st.spinner("Testing NJOY..."):
                    import subprocess

                    try:
                        result = subprocess.run(
                            [njoy_exe],
                            input=b"stop\n",
                            capture_output=True,
                            timeout=5,
                        )
                        st.success("✓ NJOY executable is accessible and responds correctly")
                        st.info(f"Return code: {result.returncode}")
                    except subprocess.TimeoutExpired:
                        st.success("✓ NJOY is running (timed out waiting for input, which is expected)")
                    except FileNotFoundError:
                        st.error(f"❌ Executable not found: {njoy_exe}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"❌ Error testing NJOY: {exc}")

with tab5:
    st.header("User Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Account Information")
        st.text_input("Name", value=current_user.get("full_name", "Guest"), disabled=True)
        st.text_input(
            "Email",
            value=current_user.get("email", "guest@kika.app"),
            disabled=True,
        )

        if current_user.get("is_guest"):
            st.info("Create an account to sync your preferences across devices.", icon="ℹ️")
        else:
            st.success("Signed in. Use the Save button below to persist changes.", icon="✅")

        st.markdown("---")
        st.subheader("Subscription")
        st.markdown("**Plan:** Free (MVP)")
        st.markdown("**Features:**")
        st.markdown(
            """
            - ✓ Basic ACE visualization
            - ✓ Cross section plotting
            - ✓ Angular distribution plotting
            - ✗ Batch processing
            - ✗ Cloud storage
            - ✗ API access
            """
        )

    with col2:
        st.subheader("Usage Statistics")
        st.metric("Files processed", "0", help="Total files analyzed")
        st.metric("Plots generated", "0", help="Total plots created")
        st.metric("Storage used", "0 MB / ∞", help="Data storage")

        st.markdown("---")
        st.subheader("Preferences")
        auto_save = st.checkbox(
            "Auto-save session",
            value=bool(profile_settings.get("auto_save", False)),
        )
        if auto_save != profile_settings.get("auto_save"):
            update_setting(("profile", "auto_save"), auto_save)

        email_notifications = st.checkbox(
            "Email notifications",
            value=bool(profile_settings.get("email_notifications", False)),
        )
        if email_notifications != profile_settings.get("email_notifications"):
            update_setting(("profile", "email_notifications"), email_notifications)

        share_analytics = st.checkbox(
            "Share anonymous usage data",
            value=bool(profile_settings.get("share_analytics", False)),
        )
        if share_analytics != profile_settings.get("share_analytics"):
            update_setting(("profile", "share_analytics"), share_analytics)


# Action buttons
st.markdown("---")
snapshot = collect_settings_snapshot()
user_id = current_user.get("id")
is_guest = current_user.get("is_guest", False)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        success, message = save_user_settings(user_id, snapshot)
        if success:
            st.success(message)
            st.balloons()
        else:
            st.warning(message)

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        defaults = reset_settings_to_defaults(user_id, is_guest)
        if not is_guest:
            save_user_settings(user_id, defaults)
            st.success("Settings reset to defaults.")
        else:
            st.info("Defaults restored for this session.")
        st.rerun()

with col3:
    st.button("📥 Import Settings", disabled=True, use_container_width=True)
    st.caption("Import support coming soon.")

with col4:
    st.download_button(
        "📤 Export Settings",
        data=json.dumps(snapshot, indent=2),
        file_name="kika_settings.json",
        mime="application/json",
        use_container_width=True,
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Settings • KIKA v0.1.0 (MVP)</p>
</div>
""",
    unsafe_allow_html=True,
)
