# Tauri Migration Progress

## ✅ Completed (Step 1)

### Environment Setup
- [x] Node.js 20 LTS installed via nvm
- [x] Rust 1.91 installed
- [x] System dependencies for Tauri (WebKit, GTK, etc.)
- [x] All build tools verified

### Project Structure
- [x] `package.json` with all required dependencies
- [x] TypeScript configuration (`tsconfig.json`)
- [x] Vite configuration for Tauri
- [x] Basic React app structure
- [x] Material-UI theme configured
- [x] Project documentation (README.md, MIGRATION_GUIDE.md)
- [x] Setup script for easy environment installation

## 🔄 Next Steps (In Progress - Step 2)

### Immediate Tasks
1. Install npm dependencies: `npm install`
2. Initialize Tauri backend: `npm run tauri init`
3. Create Tauri Rust commands for file handling
4. Test basic Tauri ↔ React communication

### File Upload Implementation
- Create Tauri command for file dialog
- Implement file reading from local filesystem
- Connect to Python backend API for processing

### Auth Integration
- Port authentication logic from Streamlit
- Create React context for auth state
- Connect to existing kika-backend

## 📋 Remaining Migration Work

### Core Features (Steps 3-8)
1. **Authentication** (3-4 hours)
   - Login/Register UI
   - Protected routes
   - Session management

2. **File Upload** (2-3 hours)
   - Native file dialogs
   - File type detection
   - Upload progress

3. **ACE Viewer** (1 week)
   - Cross section plots
   - Angular distributions
   - Multi-library comparison
   - Export functionality

4. **ENDF Viewer** (1 week)
   - MF4 data visualization
   - Uncertainty bands
   - Library comparisons

5. **NJOY Processing** (3-4 days)
   - ENDF to ACE conversion UI
   - Temperature selection
   - Process monitoring

6. **Settings** (2 days)
   - Plot configuration
   - User preferences
   - Persistent storage

7. **Build & Package** (2-3 days)
   - Windows MSI
   - Linux DEB/AppImage
   - Icons and branding

## 📊 Estimated Timeline

- **Week 1**: Setup + Auth + File handling ✅ (partially done)
- **Week 2-3**: ACE Viewer migration
- **Week 4**: ENDF Viewer migration
- **Week 5**: NJOY + Settings
- **Week 6**: Polish, testing, packaging

## 🚀 How to Continue

```bash
# 1. Install dependencies
cd /home/MONLEON-JUAN/MCNPy/kika-desktop
npm install

# 2. Initialize Tauri (if not already done)
npm run tauri init
# When prompted:
#   - App name: KIKA
#   - Window title: KIKA - Nuclear Data Viewer  
#   - Before dev command: npm run dev
#   - Before build command: npm run build
#   - Dev path: http://localhost:1420
#   - Dist dir: ../dist

# 3. Start development
npm run tauri:dev
```

## 📝 Notes

- Original Streamlit app preserved in branch `feature/streamlit-ui`
- Backend (kika-backend) remains unchanged and compatible
- MCNPy core library untouched
- All migration work in branch `feature/tauri-desktop-ui`

## ⚠️ Known Issues to Resolve

- [ ] Add @types/node to devDependencies for Vite config
- [ ] Create initial Tauri routes file
- [ ] Create AuthContext implementation
- [ ] Set up proxy configuration for local backend API

## 📚 Resources

- [Tauri Documentation](https://tauri.app/v1/guides/)
- [Material-UI Components](https://mui.com/material-ui/getting-started/)
- [React Router](https://reactrouter.com/)
- [Zustand State Management](https://github.com/pmndrs/zustand)
