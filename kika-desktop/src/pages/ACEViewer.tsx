import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Tabs,
  Tab,
  Alert,
} from '@mui/material';
import { FileUpload } from '../components/FileUpload';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export const ACEViewer: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loadedFiles, setLoadedFiles] = useState<{ name: string; path: string }[]>([]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleFileSelect = (file: { name: string; path: string; content: string }) => {
    console.log('File selected:', file);
    setLoadedFiles(prev => [...prev, { name: file.name, path: file.path }]);
    // TODO: Parse ACE file content using MCNPy
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          📊 ACE Data Viewer
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Upload and visualize ACE format nuclear data files
        </Typography>

        <Paper sx={{ mt: 3 }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Home" />
            <Tab label="Viewer" />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <Typography variant="h5" gutterBottom>
              📖 About ACE Viewer
            </Typography>

            <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: '1fr 1fr', mt: 3 }}>
              <Box>
                <Typography variant="h6" gutterBottom>
                  What is ACE?
                </Typography>
                <Typography variant="body2" paragraph>
                  ACE (A Compact ENDF) is a binary format used by Monte Carlo transport codes
                  like MCNP to store nuclear data. It contains cross sections, angular
                  distributions, and other reaction data in a compact, fast-to-access format.
                </Typography>

                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Supported Data Types:
                </Typography>
                <Box component="ul" sx={{ pl: 2 }}>
                  <li>
                    <strong>Cross Sections</strong>: Reaction probabilities vs. energy (barns)
                  </li>
                  <li>
                    <strong>Angular Distributions</strong>: Scattering angle probabilities
                  </li>
                </Box>

                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Features:
                </Typography>
                <Box component="ul" sx={{ pl: 2 }}>
                  <li>Multi-file comparison - overlay data from different libraries</li>
                  <li>Interactive configuration - customize every aspect of the plot</li>
                  <li>High-quality export - save publication-ready figures</li>
                  <li>Energy interpolation - evaluate angular distributions at any energy</li>
                </Box>
              </Box>

              <Box>
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="h6" gutterBottom>
                    Current Status
                  </Typography>
                  <Typography variant="h4" color="primary">
                    {loadedFiles.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Loaded ACE Files
                  </Typography>

                  {loadedFiles.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" fontWeight="bold">
                        Loaded files:
                      </Typography>
                      {loadedFiles.map((file, idx) => (
                        <Typography key={idx} variant="body2" sx={{ ml: 1 }}>
                          • {file.name}
                        </Typography>
                      ))}
                    </Box>
                  )}

                  {loadedFiles.length === 0 && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      Upload ACE files to get started
                    </Alert>
                  )}
                </Paper>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Getting Started
                  </Typography>
                  <Box component="ol" sx={{ pl: 2 }}>
                    <li>Upload ACE files using the file picker below</li>
                    <li>Go to the <strong>Viewer</strong> tab</li>
                    <li>Add data series and configure styling</li>
                    <li>Customize labels, scales, and appearance</li>
                    <li>Export your plot in various formats</li>
                  </Box>
                </Box>
              </Box>
            </Box>

            <Paper sx={{ p: 3, mt: 4, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
              <Typography variant="h6" gutterBottom>
                📁 Upload ACE Files
              </Typography>
              <FileUpload
                accept={['ace', '02c', '03c', '20c', '21c', '70c', '80c']}
                onFileSelect={handleFileSelect}
                label="Select an ACE format file"
                maxSizeMB={50}
              />
            </Paper>

            <Box sx={{ mt: 3, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
              <Typography variant="h6" gutterBottom>
                💡 Quick Tips
              </Typography>
              <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: 'repeat(3, 1fr)' }}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    📁 File Management
                  </Typography>
                  <Typography variant="body2">
                    ACE files are typically named with suffixes like .02c (ENDF/B-VII.1),
                    .80c (JEFF-3.3), etc.
                  </Typography>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    🔍 Comparison Mode
                  </Typography>
                  <Typography variant="body2">
                    Load multiple files to compare data from different nuclear data libraries
                    side by side.
                  </Typography>
                </Paper>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    📊 Export Options
                  </Typography>
                  <Typography variant="body2">
                    Save plots as PNG, PDF, or SVG for publications and presentations.
                  </Typography>
                </Paper>
              </Box>
            </Box>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Typography variant="h5" gutterBottom>
              🎨 Plot Configuration
            </Typography>

            {loadedFiles.length === 0 ? (
              <Alert severity="warning">
                No files loaded. Please upload ACE files from the Home tab first.
              </Alert>
            ) : (
              <Alert severity="info">
                🚧 Viewer functionality coming soon - Migration in progress
              </Alert>
            )}
          </TabPanel>
        </Paper>
      </Box>
    </Container>
  );
};
