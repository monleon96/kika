# KIKA Desktop - Nuclear Data Viewer

Modern desktop application for visualizing and analyzing nuclear data, built with Tauri + React + TypeScript.

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│  Desktop App (Tauri + React)        │
│  - Native window                     │
│  - Local file handling               │
│  - TypeScript UI                     │
└───────────┬─────────────────────────┘
            │ IPC Commands
    ┌───────▼──────────┐
    │  Tauri Core      │  (Rust)
    │  - File system   │
    │  - Security      │
    │  - Process mgmt  │
    └───────┬──────────┘
            │ HTTP
    ┌───────▼──────────┐
    │  FastAPI Local   │  (Python)
    │  - MCNPy core    │
    │  - ACE/ENDF      │
    │  - Plotting      │
    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Node.js 18+** (for React)
2. **Rust 1.70+** (for Tauri)
3. **Python 3.12+** (for backend - already installed)

### Installation

1. **Install Node.js and Rust:**

```bash
# Run the setup script
cd kika-desktop
./setup_dev_environment.sh
```

Or manually:

```bash
# Install Node.js (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

2. **Install project dependencies:**

```bash
# Frontend dependencies
npm install

# Install Tauri CLI
npm install -g @tauri-apps/cli
```

3. **Start development:**

```bash
# Terminal 1: Start Python backend
cd ../kika-backend
python app.py

# Terminal 2: Start Tauri dev
cd ../kika-desktop
npm run tauri dev
```

## 📦 Build for Distribution

```bash
# Build for current platform
npm run tauri build

# Output will be in:
# - Windows: src-tauri/target/release/bundle/msi/
# - Linux: src-tauri/target/release/bundle/deb/ or .appimage
```

### Build size expectations:
- **Windows MSI**: ~30-50MB
- **Linux AppImage**: ~40-60MB
- **Linux DEB**: ~30-45MB

Much smaller than Streamlit portable (~300-600MB)!

## 📂 Project Structure

```
kika-desktop/
├── src/                      # React frontend
│   ├── App.tsx               # Main app component
│   ├── main.tsx              # Entry point
│   ├── components/           # Reusable components
│   │   ├── FileUploader.tsx
│   │   ├── PlotViewer.tsx
│   │   └── Layout/
│   ├── pages/                # Page components
│   │   ├── ACEViewer.tsx
│   │   ├── ENDFViewer.tsx
│   │   ├── NJOYProcessor.tsx
│   │   └── Settings.tsx
│   ├── hooks/                # Custom React hooks
│   │   ├── useAuth.ts
│   │   └── useFileManager.ts
│   ├── services/             # API services
│   │   ├── api.ts            # Backend API client
│   │   └── tauri.ts          # Tauri commands
│   ├── types/                # TypeScript types
│   └── utils/                # Utilities
│
├── src-tauri/                # Tauri backend (Rust)
│   ├── src/
│   │   └── main.rs           # Tauri commands & setup
│   ├── Cargo.toml            # Rust dependencies
│   ├── tauri.conf.json       # Tauri configuration
│   └── icons/                # App icons
│
├── public/                   # Static assets
├── package.json              # Node dependencies
├── tsconfig.json             # TypeScript config
├── vite.config.ts            # Vite bundler config
└── README.md                 # This file
```

## 🔧 Development

### Available Scripts

```bash
# Start dev mode (hot reload)
npm run tauri dev

# Build for production
npm run tauri build

# Run frontend only (for UI development)
npm run dev

# Lint TypeScript
npm run lint

# Type checking
npm run type-check
```

### Tauri Commands

Custom commands available from React:

```typescript
import { invoke } from '@tauri-apps/api/tauri';

// Process ACE file
const result = await invoke('process_ace_file', { path: '/path/to/file.ace' });

// Call Python API
const plot = await invoke('generate_plot', { 
  zaid: '92235',
  mt: 2,
  library: 'JEFF-3.3'
});
```

## 🔌 API Integration

The app communicates with the local FastAPI backend on `http://localhost:8000`.

**Backend must be running** for full functionality:

```bash
cd ../kika-backend
uvicorn app:app --reload
```

## 🎨 UI Framework

- **React 18** with TypeScript
- **Material-UI (MUI)** for components
- **React Router** for navigation
- **Zustand** for state management (lightweight alternative to Redux)
- **React Query** for API data fetching
- **Plotly.js** for interactive plots

## 📋 Migration Status

Progress from Streamlit to Tauri:

- [x] Project setup & structure
- [ ] Authentication integration
- [ ] File upload system
- [ ] ACE Viewer
  - [ ] Cross sections viewer
  - [ ] Angular distributions
  - [ ] Multi-library comparison
- [ ] ENDF Viewer
  - [ ] MF4 data visualization
  - [ ] Uncertainty bands
- [ ] NJOY Processing
  - [ ] ENDF to ACE conversion
  - [ ] Temperature selection
- [ ] Settings page
- [ ] Build & packaging

## 🚨 Troubleshooting

### Tauri build fails
```bash
# Update Rust
rustup update

# Clear cache
cd src-tauri
cargo clean
```

### Frontend not connecting to backend
- Ensure Python backend is running on port 8000
- Check CORS settings in `kika-backend/app.py`
- Verify `src/services/api.ts` has correct URL

### Hot reload not working
```bash
# Restart dev server
npm run tauri dev
```

## 📖 Documentation

- [Tauri Docs](https://tauri.app/)
- [React Docs](https://react.dev/)
- [MUI Components](https://mui.com/)
- [MCNPy Documentation](../README.md)

## 🤝 Contributing

This is a migration from the Streamlit UI. See the original implementation at `../streamlit_app/`.

## 📝 License

Same as MCNPy - GNU General Public License v3.0
