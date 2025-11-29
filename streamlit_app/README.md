# KIKA - Nuclear Data Viewer

A modern web interface for visualizing and analyzing nuclear data, powered by KIKA.

![KIKA Banner](assets/banner.png)

## 🚀 Quick Start

### Installation

1. **Install dependencies:**

```bash
# From the root of KIKA repository
cd /home/MONLEON-JUAN/KIKA

# Install with UI dependencies
poetry install --with ui

# Or using pip
pip install -e .[ui]
```

2. **Run the application:**

```bash
# From the streamlit_app directory
cd streamlit_app
streamlit run app.py

# Or from project root
streamlit run streamlit_app/app.py
```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in terminal

## ✨ Features

### Current (MVP v0.1.0)

- **📊 ACE Data Viewer**
  - Upload multiple ACE format files
  - Visualize cross sections
  - Plot angular distributions
  - Compare multiple nuclear data libraries (JEFF, JENDL, ENDF/B)
  - Export high-quality plots (PNG, PDF, SVG)
  - Interactive plot configuration

- **⚙️ Settings**
  - Customizable plot defaults
  - Export preferences
  - Theme configuration (coming soon)

### Coming Soon

- **📈 ENDF Data Viewer** - Full ENDF-6 format support
- **🔥 Covariance Analysis** - Uncertainty and correlation matrices
- **👤 User Authentication** - Personal accounts and saved sessions
- **☁️ Cloud Storage** - Save and share your analyses
- **📦 Batch Processing** - Process multiple files automatically
- **🔌 API Access** - Programmatic access to visualizations

## 📖 User Guide

### Uploading Files

1. Navigate to **ACE Viewer** from the home page
2. Click "Browse files" in the sidebar
3. Select one or more ACE files
4. Files will be automatically loaded and validated

### Creating Plots

#### Cross Sections

1. Select "Cross Section" as data type
2. Choose files to compare
3. Enter MT number (e.g., 2 for elastic scattering)
4. Configure plot styling in the "Styling" tab
5. Click "Generate Plot"
6. Download using the download button

#### Angular Distributions

1. Select "Angular Distribution" as data type
2. Choose files to compare
3. Enter MT number and energy (MeV)
4. Configure plot styling
5. Generate and download

### Keyboard Shortcuts

- `Ctrl + S` - Quick save plot
- `Ctrl + R` - Refresh data
- `Ctrl + /` - Toggle sidebar

## 🏗️ Project Structure

```
streamlit_app/
├── app.py                 # Main application entry point
├── pages/                 # Multi-page app pages
│   ├── 1_📊_ACE_Viewer.py
│   ├── 2_📈_ENDF_Viewer.py  (coming soon)
│   └── 3_⚙️_Settings.py
├── components/            # Reusable UI components
│   ├── file_uploader.py   (coming soon)
│   ├── plot_viewer.py     (coming soon)
│   └── data_table.py      (coming soon)
├── utils/                 # Utility functions
│   ├── session_state.py   # Session management
│   └── file_utils.py      # File handling
├── assets/                # Static assets (images, etc.)
└── README.md             # This file
```

## 🎨 Customization

### Theme

Edit `.streamlit/config.toml` (create if it doesn't exist):

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"
font = "sans serif"
```

### Plot Defaults

Configure default plot settings in the Settings page or modify `utils/plot_config.py`.

## 🐛 Troubleshooting

### Import errors

If you see `ModuleNotFoundError: No module named 'kika'`:

```bash
# Make sure you're running from the correct directory
cd /home/MONLEON-JUAN/KIKA
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run streamlit_app/app.py
```

### Port already in use

```bash
# Run on a different port
streamlit run app.py --server.port 8502
```

### Files not loading

- Check file format (should be ACE format)
- Verify file permissions
- Check browser console for errors

## 🔧 Development

### Running in development mode

```bash
# Enable auto-reload
streamlit run app.py --server.runOnSave true

# Enable debug mode
streamlit run app.py --logger.level debug
```

### Adding a new page

1. Create file in `pages/` with format: `N_emoji_Name.py`
2. Pages are automatically discovered by Streamlit
3. Use consistent styling from other pages

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Type checking
mypy streamlit_app/
```

## 📊 Technical Stack

- **Frontend:** Streamlit 1.40+
- **Backend:** KIKA (Python 3.12+)
- **Plotting:** Matplotlib + PlotBuilder API
- **Data:** NumPy, Pandas
- **Future:** React + FastAPI (Phase 2)

## 🤝 Contributing

This is currently in MVP phase. Contributions welcome after initial release!

## 📝 License

Same as KIKA - GNU General Public License v3.0

## 🔗 Links

- [KIKA GitHub](https://github.com/monleon96/KIKA)
- [KIKA Documentation](https://kika.readthedocs.io)
- [Streamlit Documentation](https://docs.streamlit.io)

## 📧 Contact

For questions or feedback:
- Email: juanmonleon96@gmail.com
- GitHub Issues: https://github.com/monleon96/KIKA/issues

---

**KIKA** v0.1.0 (MVP) • Built with ❤️ by Juan Antonio Monleon
