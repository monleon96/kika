import { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Button,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { checkMCNPyHealth } from '../services/mcnpyService';

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const [mcnpyStatus, setMcnpyStatus] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    checkBackend();
  }, []);

  const checkBackend = async () => {
    try {
      const isHealthy = await checkMCNPyHealth();
      setMcnpyStatus(isHealthy);
    } catch (error) {
      console.error('Error checking MCNPy server:', error);
      setMcnpyStatus(false);
    } finally {
      setChecking(false);
    }
  };

  return (
    <Box sx={{ width: '100%', px: 3 }}>
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Welcome to KIKA
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph>
          Nuclear Data Visualization & Analysis Platform
        </Typography>

        {/* MCNPy Server Status */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            {checking ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress size={20} />
                <Typography>Checking MCNPy server connection...</Typography>
              </Box>
            ) : mcnpyStatus ? (
              <Alert severity="success">
                ✓ MCNPy Server is running and healthy (http://localhost:8001)
              </Alert>
            ) : (
              <Alert severity="warning">
                ⚠️ MCNPy Server is not responding. Make sure the server is running:
                <Box component="code" sx={{ display: 'block', mt: 1, p: 1, bgcolor: 'rgba(0,0,0,0.1)', borderRadius: 1, fontSize: '0.85em' }}>
                  cd MCNPy && ./start-dev.sh
                </Box>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Feature Cards */}
        <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
          Available Features
        </Typography>

        <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 ACE Data Viewer
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Upload and visualize ACE format nuclear data files with cross sections
                and angular distributions.
              </Typography>
              <Button
                variant="contained"
                fullWidth
                onClick={() => navigate('/ace-viewer')}
                sx={{ mt: 1 }}
              >
                Open ACE Viewer →
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 ENDF Data Viewer
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Explore ENDF-6 format evaluated nuclear data with uncertainty bands
                and library comparisons.
              </Typography>
              <Button
                variant="contained"
                fullWidth
                onClick={() => navigate('/endf-viewer')}
                sx={{ mt: 1 }}
              >
                Open ENDF Viewer →
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🔧 NJOY Processing
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Generate ACE files from ENDF data with temperature selection and
                automatic versioning.
              </Typography>
              <Typography variant="caption" color="warning.main" sx={{ mt: 2, display: 'block' }}>
                🚧 Coming soon - Migration in progress
              </Typography>
            </CardContent>
          </Card>
        </Box>

        <Box sx={{ mt: 4, p: 3, bgcolor: 'info.light', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom>
            🎉 New Desktop Application
          </Typography>
          <Typography variant="body2">
            You're now using the new Tauri-based desktop application! This version provides:
          </Typography>
          <Box component="ul" sx={{ mt: 1 }}>
            <li>Native desktop performance</li>
            <li>Smaller file size (~40MB vs 300-600MB)</li>
            <li>No admin rights required</li>
            <li>Works offline</li>
            <li>Auto-updates support</li>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            The Streamlit version is still available in the repository on branch feature/streamlit-ui
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};
