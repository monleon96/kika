import React, { useMemo, useState, useRef, useCallback } from 'react';
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
import { useFileWorkspace } from '../contexts/FileWorkspaceContext';
import type { ENDFMetadata, WorkspaceFile } from '../types/file';
import { ENDFPlotViewer, type LoadedENDFFile } from '../components/ENDFPlotViewer';

const isTauri = '__TAURI__' in window;

const isENDFMetadata = (metadata?: WorkspaceFile['metadata']): metadata is ENDFMetadata => {
  return Boolean(metadata && 'angular_mts' in metadata);
};

const FileUploadPrompt: React.FC = () => {
  const { addFiles } = useFileWorkspace();
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (fileList: FileList) => {
      setUploading(true);
      try {
        const filesArray = Array.from(fileList);
        const filesData = await Promise.all(
          filesArray.map(async (file) => ({
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
    },
    [addFiles]
  );

  const handleTauriFileSelect = useCallback(async () => {
    setUploading(true);
    try {
      const { open: openDialog } = await import('@tauri-apps/api/dialog');
      const { readTextFile } = await import('@tauri-apps/api/fs');

      const selected = await openDialog({
        multiple: true,
        filters: [
          {
            name: 'ENDF Files',
            extensions: ['endf', 'endf6', 'txt', 'tendl', 'dat'],
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

  const handleDrag = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.type === 'dragenter' || event.type === 'dragover') {
      setDragActive(true);
    } else if (event.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      event.stopPropagation();
      setDragActive(false);
      if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
        handleFiles(event.dataTransfer.files);
      }
    },
    [handleFiles]
  );

  const handleFileInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (event.target.files && event.target.files.length > 0) {
        handleFiles(event.target.files);
      }
    },
    [handleFiles]
  );

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
        border: '3px dashed',
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
        accept=".endf,.endf6,.txt,.dat"
        onChange={handleFileInputChange}
        style={{ display: 'none' }}
      />

      <Folder sx={{ fontSize: 80, color: 'grey.400', mb: 2 }} />

      <Typography variant="h5" gutterBottom>
        No ENDF Files Loaded
      </Typography>

      <Typography variant="body1" color="text.secondary" paragraph>
        Upload ENDF-6 data to start configuring plots.
      </Typography>

      <Button
        variant="contained"
        size="large"
        startIcon={uploading ? <CircularProgress size={20} /> : <UploadFile />}
        onClick={openFileDialog}
        disabled={uploading}
        sx={{ mt: 2 }}
      >
        {uploading ? 'Uploading...' : 'Upload ENDF Files'}
      </Button>

      <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 2 }}>
        Drop files here or click to browse • Supported: .endf, .endf6, .txt
      </Typography>
    </Paper>
  );
};

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
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

export const ENDFViewer: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const { getFilesByType } = useFileWorkspace();
  const workspaceFiles = getFilesByType('endf');

  const endfFiles: LoadedENDFFile[] = useMemo(
    () =>
      workspaceFiles
        .filter((file) => file.status === 'ready' && isENDFMetadata(file.metadata))
        .map((file) => ({
          id: file.id,
          name: file.name,
          path: file.path,
          content: file.content,
          metadata: file.metadata as ENDFMetadata,
        })),
    [workspaceFiles]
  );

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box sx={{ width: '100%', px: 3 }}>
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          📈 ENDF Data Viewer
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Upload and visualize ENDF-6 angular distributions and their uncertainties.
        </Typography>

        <Paper sx={{ mt: 3 }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Home" />
            <Tab label="Viewer" />
          </Tabs>

          <TabPanel value={tabValue} index={0}>
            <Typography variant="h5" gutterBottom>
              📖 About ENDF Viewer
            </Typography>
            <Box sx={{ display: 'grid', gap: 3, gridTemplateColumns: { sm: '1fr', md: '1fr 1fr' }, mt: 3 }}>
              <Box>
                <Typography variant="h6" gutterBottom>
                  What is ENDF?
                </Typography>
                <Typography variant="body2" paragraph>
                  ENDF (Evaluated Nuclear Data File) stores reaction cross sections, angular distributions,
                  and covariance data used in transport and reactor calculations. This viewer focuses on
                  MF4 (angular Legendre coefficients) and MF34 (angular covariances).
                </Typography>
                <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                  Features
                </Typography>
                <Box component="ul" sx={{ pl: 2 }}>
                  <li>Compare angular distributions across multiple libraries</li>
                  <li>Visualize MF34 uncertainty bands on top of MF4 data</li>
                  <li>Customize axes, legend, and styling for publication-ready plots</li>
                  <li>Export high-quality Matplotlib figures directly from the desktop app</li>
                </Box>
              </Box>
              <Box>
                <Paper sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="h6" gutterBottom>
                    Current Status
                  </Typography>
                  <Typography variant="h4" color="primary">
                    {endfFiles.length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Loaded ENDF Files
                  </Typography>
                  {endfFiles.length > 0 ? (
                    <Box sx={{ mt: 2 }}>
                      {endfFiles.slice(0, 5).map((file) => (
                        <Box key={file.id} sx={{ mb: 1 }}>
                          <Typography variant="body2">• {file.name}</Typography>
                          {file.metadata?.isotope && (
                            <Chip label={file.metadata.isotope} size="small" sx={{ mt: 0.5 }} />
                          )}
                        </Box>
                      ))}
                      {endfFiles.length > 5 && (
                        <Typography variant="caption" color="text.secondary">
                          +{endfFiles.length - 5} more
                        </Typography>
                      )}
                    </Box>
                  ) : (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      Open the 📁 File Workspace to upload ENDF files.
                    </Alert>
                  )}
                </Paper>
                <Box sx={{ mt: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Getting Started
                  </Typography>
                  <Box component="ol" sx={{ pl: 2 }}>
                    <li>Open the File Workspace (📁 icon in the top bar)</li>
                    <li>Upload ENDF-6 files containing MF4/MF34 data</li>
                    <li>Switch to the Viewer tab</li>
                    <li>Add series, choose MT/Legendre order, and customize styling</li>
                    <li>Export high-quality plots when ready</li>
                  </Box>
                </Box>
              </Box>
            </Box>
            <Paper sx={{ p: 3, mt: 4, bgcolor: 'info.light' }}>
              <Typography variant="h6" gutterBottom>
                📁 File Workspace
              </Typography>
              <Typography variant="body2">
                Files are managed globally through the File Workspace (left drawer). Upload once and reuse the same
                files across ACE, ENDF, and future viewers.
              </Typography>
            </Paper>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Typography variant="h5" gutterBottom>
              🎨 Plot Viewer
            </Typography>
            {endfFiles.length === 0 ? <FileUploadPrompt /> : <ENDFPlotViewer files={endfFiles} />}
          </TabPanel>
        </Paper>
      </Box>
    </Box>
  );
};
