import { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Alert,
  CircularProgress,
} from '@mui/material';
import { invoke } from '@tauri-apps/api/tauri';

export const Home: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    checkBackend();
  }, []);

  const checkBackend = async () => {
    try {
      const isHealthy = await invoke<boolean>('check_backend_health');
      setBackendStatus(isHealthy);
    } catch (error) {
      console.error('Error checking backend:', error);
      setBackendStatus(false);
    } finally {
      setChecking(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Welcome to KIKA
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph>
          Nuclear Data Visualization & Analysis Platform
        </Typography>

        {/* Backend Status */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            {checking ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CircularProgress size={20} />
                <Typography>Checking backend connection...</Typography>
              </Box>
            ) : backendStatus ? (
              <Alert severity="success">
                ✓ Backend is running and healthy (http://localhost:8000)
              </Alert>
            ) : (
              <Alert severity="error">
                ✗ Backend is not responding. Please start the Python backend:
                <Box component="pre" sx={{ mt: 1, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                  cd ../kika-backend{'\n'}
                  python app.py
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
              <Typography variant="body2" color="text.secondary">
                Upload and visualize ACE format nuclear data files with cross sections
                and angular distributions.
              </Typography>
              <Typography variant="caption" color="warning.main" sx={{ mt: 2, display: 'block' }}>
                🚧 Coming soon - Migration in progress
              </Typography>
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
              <Typography variant="caption" color="warning.main" sx={{ mt: 2, display: 'block' }}>
                🚧 Coming soon - Migration in progress
              </Typography>
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
    </Container>
  );
};
