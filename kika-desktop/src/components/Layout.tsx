import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Box, AppBar, Toolbar, Typography, IconButton, Button } from '@mui/material';
import { Logout, Home as HomeIcon } from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';

export const Layout: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
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

          <Typography variant="body2" sx={{ mr: 2 }}>
            {user?.email}
          </Typography>
          <IconButton color="inherit" onClick={logout} title="Logout">
            <Logout />
          </IconButton>
        </Toolbar>
      </AppBar>
      
      <Box component="main" sx={{ flexGrow: 1, p: 3, overflow: 'auto' }}>
        <Outlet />
      </Box>
    </Box>
  );
};
