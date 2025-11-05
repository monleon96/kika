# Migration Guide: Streamlit to Tauri + React

This document explains the key differences and patterns when migrating from Streamlit to Tauri + React.

## 🎯 Key Architectural Differences

### Streamlit
- **Server-based**: Runs a Python web server
- **Auto-reload**: Full page refresh on interaction
- **State**: Session-based, server-side
- **Files**: Uploaded to server memory
- **Distribution**: Python runtime required

### Tauri + React
- **Desktop native**: Standalone application
- **Reactive**: Component-level updates
- **State**: Client-side, persistent
- **Files**: Local file system access
- **Distribution**: Single executable (~40MB)

---

## 📋 Migration Patterns

### 1. File Upload

#### Before (Streamlit):
```python
uploaded_file = st.file_uploader("Upload ACE file", type=['ace'])

if uploaded_file:
    ace = ACE(uploaded_file)
    st.success(f"Loaded {uploaded_file.name}")
```

#### After (React + Tauri):
```typescript
import { open } from '@tauri-apps/api/dialog';
import { invoke } from '@tauri-apps/api/tauri';

const handleFileUpload = async () => {
  // Open native file dialog
  const selected = await open({
    filters: [{
      name: 'ACE Files',
      extensions: ['ace', '02c', '20c']
    }]
  });
  
  if (selected) {
    // Process file via Tauri command
    const result = await invoke('process_ace_file', {
      path: selected as string
    });
    
    // Update state
    setAceData(result);
  }
};

return (
  <Button onClick={handleFileUpload}>
    Upload ACE File
  </Button>
);
```

**Key Tauri command** (in `src-tauri/src/main.rs`):
```rust
#[tauri::command]
async fn process_ace_file(path: String) -> Result<String, String> {
    // Call Python backend
    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:8000/api/process-ace")
        .json(&serde_json::json!({"path": path}))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    let data = res.text().await.map_err(|e| e.to_string())?;
    Ok(data)
}
```

---

### 2. State Management

#### Before (Streamlit):
```python
# Session state
if 'ace_files' not in st.session_state:
    st.session_state.ace_files = {}

st.session_state.ace_files[filename] = ace_obj
```

#### After (React with Zustand):
```typescript
// stores/fileStore.ts
import create from 'zustand';
import { persist } from 'zustand/middleware';

interface FileStore {
  aceFiles: Record<string, ACEData>;
  addAceFile: (name: string, data: ACEData) => void;
  removeAceFile: (name: string) => void;
}

export const useFileStore = create<FileStore>()(
  persist(
    (set) => ({
      aceFiles: {},
      addAceFile: (name, data) =>
        set((state) => ({
          aceFiles: { ...state.aceFiles, [name]: data }
        })),
      removeAceFile: (name) =>
        set((state) => {
          const { [name]: _, ...rest } = state.aceFiles;
          return { aceFiles: rest };
        }),
    }),
    {
      name: 'kika-file-storage', // LocalStorage key
    }
  )
);

// Usage in component
const { aceFiles, addAceFile } = useFileStore();

const handleUpload = async (path: string) => {
  const data = await invoke('process_ace_file', { path });
  addAceFile(path, data);
};
```

---

### 3. Plotting

#### Before (Streamlit):
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(energy, xs)
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Cross Section (barns)')

st.pyplot(fig)
```

#### After (React - Option 1: Server-side rendering):
```typescript
// Request plot from Python backend
const fetchPlot = async () => {
  const response = await fetch('http://localhost:8000/api/plot/xs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      zaid: '92235',
      mt: 2,
      energy_range: [1e-5, 20]
    })
  });
  
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  setPlotUrl(url);
};

return <img src={plotUrl} alt="Cross section plot" />;
```

#### After (React - Option 2: Client-side with Plotly):
```typescript
import Plot from 'react-plotly.js';

const PlotViewer: React.FC<{data: PlotData}> = ({ data }) => {
  return (
    <Plot
      data={[
        {
          x: data.energy,
          y: data.xs,
          type: 'scatter',
          mode: 'lines',
          name: 'Cross Section'
        }
      ]}
      layout={{
        title: 'Cross Section vs Energy',
        xaxis: { 
          title: 'Energy (MeV)',
          type: 'log'
        },
        yaxis: { 
          title: 'Cross Section (barns)',
          type: 'log'
        }
      }}
    />
  );
};
```

**Backend endpoint** (FastAPI):
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import io

@app.post("/api/plot/xs")
async def plot_cross_section(request: PlotRequest):
    # Generate plot
    fig, ax = plt.subplots()
    ace = ACE(request.zaid)
    ace.plot_xs(MT=request.mt, ax=ax)
    
    # Return as image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")
```

---

### 4. Multi-page Navigation

#### Before (Streamlit):
```python
# Automatic - just create pages/1_Page.py

# Navigate programmatically
if st.button("Go to ACE Viewer"):
    st.switch_page("pages/1_ACE_Viewer.py")
```

#### After (React Router):
```typescript
// App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import ACEViewer from './pages/ACEViewer';
import ENDFViewer from './pages/ENDFViewer';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/ace-viewer" element={<ACEViewer />} />
        <Route path="/endf-viewer" element={<ENDFViewer />} />
      </Routes>
    </BrowserRouter>
  );
}

// Navigate programmatically
import { useNavigate } from 'react-router-dom';

const navigate = useNavigate();
<Button onClick={() => navigate('/ace-viewer')}>
  Go to ACE Viewer
</Button>
```

---

### 5. Authentication

#### Before (Streamlit):
```python
from utils.backend_auth import require_user

current_user = require_user()
st.write(f"Welcome, {current_user['email']}")
```

#### After (React with Context):
```typescript
// contexts/AuthContext.tsx
import { createContext, useContext, useState } from 'react';

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  
  const login = async (email: string, password: string) => {
    const response = await fetch('http://localhost:8000/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    
    if (response.ok) {
      const userData = await response.json();
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
    }
  };
  
  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };
  
  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

// Usage
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};

// In component
const { user } = useAuth();
if (!user) return <LoginPage />;
```

---

### 6. Sidebar

#### Before (Streamlit):
```python
with st.sidebar:
    st.title("KIKA")
    st.markdown("---")
    
    if st.button("ACE Viewer"):
        st.switch_page("pages/1_ACE_Viewer.py")
```

#### After (React with MUI):
```typescript
import { Drawer, List, ListItem, ListItemButton, ListItemIcon } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  
  return (
    <Drawer
      variant="permanent"
      sx={{ width: 240 }}
    >
      <List>
        <ListItem>
          <Typography variant="h5">KIKA</Typography>
        </ListItem>
        
        <ListItemButton onClick={() => navigate('/ace-viewer')}>
          <ListItemIcon>📊</ListItemIcon>
          <ListItemText primary="ACE Viewer" />
        </ListItemButton>
        
        <ListItemButton onClick={() => navigate('/endf-viewer')}>
          <ListItemIcon>📈</ListItemIcon>
          <ListItemText primary="ENDF Viewer" />
        </ListItemButton>
      </List>
    </Drawer>
  );
};
```

---

## 🔄 Data Flow Comparison

### Streamlit Flow
```
User Action → Full Page Reload → Python Re-execution → HTML Re-render
```

### Tauri + React Flow
```
User Action → React Component Update → Tauri Command (if needed) → 
Python API Call (if needed) → Update State → Re-render Component Only
```

**Advantages:**
- ⚡ Faster (only updates changed components)
- 💾 Persistent state (survives page refresh)
- 🔌 Offline capable (except API calls)

---

## 📦 Deployment Comparison

### Streamlit
```bash
# User needs:
- Python 3.12
- All dependencies installed
- Run: streamlit run app.py

# Distribution:
- Share source code + requirements.txt
- Or: 600MB portable Python bundle
```

### Tauri
```bash
# User needs:
- Nothing! Just download and run

# Distribution:
- Windows: .msi installer (40MB)
- Linux: .deb or .AppImage (45MB)
- Auto-updater support

# Build:
npm run tauri build
```

---

## 🎨 UI Component Mapping

| Streamlit | React + MUI Equivalent |
|-----------|------------------------|
| `st.button()` | `<Button>` |
| `st.selectbox()` | `<Select>` |
| `st.slider()` | `<Slider>` |
| `st.text_input()` | `<TextField>` |
| `st.checkbox()` | `<Checkbox>` |
| `st.radio()` | `<RadioGroup>` |
| `st.multiselect()` | `<Autocomplete multiple>` |
| `st.file_uploader()` | `<Button>` + Tauri file dialog |
| `st.expander()` | `<Accordion>` |
| `st.tabs()` | `<Tabs>` |
| `st.columns()` | `<Grid>` or `<Stack>` |
| `st.success()` | `<Alert severity="success">` |
| `st.error()` | `<Alert severity="error">` |
| `st.info()` | `<Alert severity="info">` |

---

## 🐛 Common Pitfalls

### 1. Async/Await
**Streamlit**: Synchronous by default
**React**: Async operations need proper handling

```typescript
// ❌ Wrong
const loadFile = () => {
  const data = invoke('load_file'); // Returns Promise
  setData(data); // undefined!
};

// ✅ Correct
const loadFile = async () => {
  const data = await invoke('load_file');
  setData(data);
};

// ✅ Or with useEffect
useEffect(() => {
  invoke('load_file').then(setData);
}, []);
```

### 2. State Updates
**Streamlit**: Immediate
**React**: Batched and asynchronous

```typescript
// ❌ Wrong
setCount(count + 1);
console.log(count); // Still old value!

// ✅ Correct
setCount(prev => prev + 1);
// Or use useEffect to react to changes
useEffect(() => {
  console.log(count); // Updated value
}, [count]);
```

### 3. File Paths
**Streamlit**: Handles as file-like objects
**Tauri**: Needs absolute file system paths

```typescript
// Get file path from dialog
const path = await open();

// ❌ Wrong - sending File object
invoke('process_file', { file: fileObject });

// ✅ Correct - sending path
invoke('process_file', { path: path as string });
```

---

## 📚 Learning Resources

- **Tauri**: https://tauri.app/v1/guides/
- **React**: https://react.dev/learn
- **TypeScript**: https://www.typescriptlang.org/docs/
- **Material-UI**: https://mui.com/material-ui/getting-started/
- **React Router**: https://reactrouter.com/
- **Zustand**: https://github.com/pmndrs/zustand

---

## 🚀 Next Steps

1. Review this guide
2. Run `./setup_dev_environment.sh`
3. Initialize project with `npm create tauri-app`
4. Start with authentication (existing backend)
5. Migrate file upload system
6. Port one page at a time (start with ACE Viewer)

Questions? Check the main README.md or open an issue.
