import React, { useState, useMemo, useRef, useCallback } from 'react';
import {
  Typography,
  Box,
  Paper,
  Tabs,
  Tab,
  Alert,
  Chip,
  Button,
  CircularProgress,
} from '@mui/material';
import { UploadFile, Folder } from '@mui/icons-material';
import { PlotViewer, type LoadedACEFile } from '../components/PlotViewer';
import { useFileWorkspace } from '../contexts/FileWorkspaceContext';

// Check if running in Tauri
const isTauri = '__TAURI__' in window;

// File upload prompt component
const FileUploadPrompt: React.FC = () => {
  const { addFiles } = useFileWorkspace();
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(async (fileList: FileList) => {
    setUploading(true);
    try {
      const filesArray = Array.from(fileList);
      const filesData = await Promise.all(
        filesArray.map(async file => ({
          name: file.name,
          path: file.name,
          content: await file.text(),
        }))
      );
      await addFiles(filesData);
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setUploading(false);
    }
  }, [addFiles]);

  const handleTauriFileSelect = useCallback(async () => {
    setUploading(true);
    try {
      const { open: openDialog } = await import('@tauri-apps/api/dialog');
      const { readTextFile } = await import('@tauri-apps/api/fs');

      const selected = await openDialog({
        multiple: true,
        filters: [
          {
            name: 'Nuclear Data Files',
            extensions: ['ace', 'endf', 'endf6', '02c', '03c', '20c', '21c', '70c', '80c'],
          },
        ],
      });

      if (!selected || !Array.isArray(selected) || selected.length === 0) {
        setUploading(false);
        return;
      }

      const filesData = await Promise.all(
        selected.map(async (filePath: string) => {
          const fileName = filePath.split(/[\\/]/).pop() || 'unknown';
          const content = await readTextFile(filePath);
          return { name: fileName, path: filePath, content };
        })
      );

      await addFiles(filesData);
    } catch (error) {
      console.error('Error selecting files:', error);
    } finally {
      setUploading(false);
    }
  }, [addFiles]);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  }, [handleFiles]);

  const openFileDialog = useCallback(() => {
    if (isTauri) {
      handleTauriFileSelect();
    } else {
      fileInputRef.current?.click();
    }
  }, [handleTauriFileSelect]);

  return (
    <Paper
      sx={{
        p: 6,
        textAlign: 'center',
        bgcolor: dragActive ? 'action.hover' : 'background.paper',
        border: dragActive ? '3px dashed' : '3px dashed',
        borderColor: dragActive ? 'primary.main' : 'grey.300',
        transition: 'all 0.2s',
      }}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".ace,.endf,.endf6,.02c,.03c,.20c,.21c,.70c,.80c"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      <Folder sx={{ fontSize: 80, color: 'grey.400', mb: 2 }} />
      
      <Typography variant="h5" gutterBottom>
        No ACE Files Loaded
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload ACE files to start visualizing nuclear data
      </Typography>

      <Button
        variant="contained"
        size="large"
        startIcon={uploading ? <CircularProgress size={20} /> : <UploadFile />}
        onClick={openFileDialog}
        disabled={uploading}
        sx={{ mt: 2 }}
      >
        {uploading ? 'Uploading...' : 'Upload ACE Files'}
      </Button>

      <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 2 }}>
        Drop files here or click to browse
      </Typography>

      <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
        Supported formats: .ace, .02c, .03c, .20c, .21c, .70c, .80c
      </Typography>
    </Paper>
  );
};

// TabPanel component
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = (props) => {
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
};

export const ACEViewer: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const { getFilesByType } = useFileWorkspace();

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Convert workspace files to LoadedACEFile format
  const loadedFiles: LoadedACEFile[] = useMemo(() => {
    const aceFiles = getFilesByType('ace');
    return aceFiles.map(file => ({
      name: file.name,
      path: file.path,
      content: file.content,
      info: file.metadata && 'atomic_weight_ratio' in file.metadata ? file.metadata : undefined,
      error: file.error,
    }));
  }, [getFilesByType]);

  return (
    <Box sx={{ width: '100%', px: 3 }}>
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
                      <Typography variant="body2" fontWeight="bold" gutterBottom>
                        Loaded files:
                      </Typography>
                      {loadedFiles.map((file, idx) => (
                        <Box key={idx} sx={{ ml: 1, mb: 1 }}>
                          <Typography variant="body2">
                            • {file.name}
                          </Typography>
                          {file.info && (
                            <Box sx={{ ml: 2, display: 'flex', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
                              <Chip label={file.info.zaid} size="small" color="primary" />
                              <Chip label={`${file.info.temperature.toFixed(1)} K`} size="small" />
                              <Chip label={`${file.info.available_reactions.length} reactions`} size="small" />
                            </Box>
                          )}
                          {file.error && (
                            <Alert severity="error" sx={{ mt: 0.5, py: 0 }}>
                              {file.error}
                            </Alert>
                          )}
                        </Box>
                      ))}
                    </Box>
                  )}

                  {loadedFiles.length === 0 && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      Open the File Workspace (📁 icon in top bar) to upload ACE files
                    </Alert>
                  )}
                </Paper>

                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Getting Started
                  </Typography>
                  <Box component="ol" sx={{ pl: 2 }}>
                    <li>Open the <strong>File Workspace</strong> from the top bar (📁 icon)</li>
                    <li>Upload ACE files (they will be available across all tabs)</li>
                    <li>Go to the <strong>Viewer</strong> tab</li>
                    <li>Add data series and configure styling</li>
                    <li>Customize labels, scales, and appearance</li>
                    <li>Export your plot in various formats</li>
                  </Box>
                </Box>
              </Box>
            </Box>

            <Paper sx={{ p: 3, mt: 4, bgcolor: 'info.light' }}>
              <Typography variant="h6" gutterBottom>
                📁 File Workspace
              </Typography>
              <Typography variant="body2">
                Files are now managed in the global File Workspace. Click the 📁 folder icon
                in the top navigation bar to open the workspace and upload ACE or ENDF files.
                All uploaded files will be available across all viewers and tabs!
              </Typography>
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
              🎨 Plot Viewer
            </Typography>

            {loadedFiles.length === 0 ? (
              <FileUploadPrompt />
            ) : (
              <PlotViewer files={loadedFiles} />
            )}
          </TabPanel>
        </Paper>
      </Box>
    </Box>
  );
};
