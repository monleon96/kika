import { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, IconButton, Button, Badge } from '@mui/material';
import { Logout, Home as HomeIcon, Folder, FolderOpen } from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import { useFileWorkspace } from '../contexts/FileWorkspaceContext';
import { FileWorkspace } from './FileWorkspace';

const WORKSPACE_WIDTH = 400;

export const Layout: React.FC = () => {
  const { user, logout } = useAuth();
  const { files } = useFileWorkspace();
  const navigate = useNavigate();
  const location = useLocation();
  const [workspaceOpen, setWorkspaceOpen] = useState(false);

  const fileCount = files.filter(f => f.status === 'ready').length;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            onClick={() => setWorkspaceOpen(!workspaceOpen)}
            title="Toggle File Workspace"
            sx={{ mr: 2 }}
          >
            <Badge badgeContent={fileCount} color="secondary">
              {workspaceOpen ? <FolderOpen /> : <Folder />}
            </Badge>
          </IconButton>

          <IconButton
            color="inherit"
            onClick={() => navigate('/')}
            title="Home"
            sx={{ mr: 2 }}
          >
            <HomeIcon />
          </IconButton>

          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            ⚛️ KIKA
          </Typography>

          <Button
            color="inherit"
            onClick={() => navigate('/ace-viewer')}
            sx={{
              mr: 1,
              bgcolor: location.pathname === '/ace-viewer' ? 'rgba(255,255,255,0.1)' : 'transparent',
            }}
          >
            📊 ACE Viewer
          </Button>
          <Button
            color="inherit"
            onClick={() => navigate('/endf-viewer')}
            sx={{
              mr: 1,
              bgcolor: location.pathname === '/endf-viewer' ? 'rgba(255,255,255,0.1)' : 'transparent',
            }}
          >
            📈 ENDF Viewer
          </Button>

          <Typography variant="body2" sx={{ mr: 2 }}>
            {user?.is_guest ? '🚀 Guest' : user?.email}
          </Typography>
          <IconButton 
            color="inherit" 
            onClick={logout} 
            title={user?.is_guest ? 'Exit Guest Mode' : 'Logout'}
          >
            <Logout />
          </IconButton>
        </Toolbar>
      </AppBar>
      
      <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden', pt: 8 }}>
        <Box
          sx={{
            width: workspaceOpen ? `${WORKSPACE_WIDTH}px` : 0,
            transition: 'width 0.3s ease-in-out',
            flexShrink: 0,
            overflow: 'hidden',
          }}
        >
          <FileWorkspace 
            open={workspaceOpen} 
            onClose={() => setWorkspaceOpen(false)}
            width={WORKSPACE_WIDTH}
          />
        </Box>
        
        <Box 
          component="main" 
          sx={{ 
            flexGrow: 1, 
            p: 3, 
            overflow: 'auto',
            width: '100%',
          }}
        >
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
};
