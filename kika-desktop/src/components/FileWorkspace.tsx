import React, { useState, useRef, useCallback } from 'react';
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Divider,
  Alert,
  CircularProgress,
  Badge,
} from '@mui/material';
import {
  Close,
  UploadFile,
  Delete,
  DeleteSweep,
  Science,
  InsertDriveFile,
  Search,
  Sort,
  FilterList,
  CheckCircle,
  Error as ErrorIcon,
  HourglassEmpty,
} from '@mui/icons-material';
import { useFileWorkspace } from '../contexts/FileWorkspaceContext';
import type { FileType, WorkspaceFile, ENDFMetadata } from '../types/file';
import { formatFileSize } from '../utils/fileDetection';

// Check if running in Tauri
const isTauri = '__TAURI__' in window;

const isENDFMetadata = (metadata?: WorkspaceFile['metadata']): metadata is ENDFMetadata => {
  return Boolean(metadata && 'angular_mts' in metadata);
};

interface FileWorkspaceProps {
  open: boolean;
  onClose: () => void;
  width?: number;
}

export const FileWorkspace: React.FC<FileWorkspaceProps> = ({ 
  open, 
  onClose,
  width = 400,
}) => {
  const {
    files,
    filter,
    addFiles,
    removeFile,
    clearAll,
    setFilter,
    getFilteredFiles,
  } = useFileWorkspace();

  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const filteredFiles = getFilteredFiles();
  const aceCount = files.filter(f => f.type === 'ace' && f.status === 'ready').length;
  const endfCount = files.filter(f => f.type === 'endf' && f.status === 'ready').length;

  // Handle file upload
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

  // Handle Tauri file dialog
  const handleTauriFileSelect = useCallback(async () => {
    setUploading(true);
    try {
      const { open: openDialog } = await import('@tauri-apps/api/dialog');
      const { readTextFile } = await import('@tauri-apps/api/fs');

      const selected = await openDialog({
        multiple: true,
        filters: [
          {
            name: 'All Files',
            extensions: ['*'],
          },
          {
            name: 'Nuclear Data Files',
            extensions: ['ace', 'endf', 'endf6', '02c', '03c', '20c', '21c', '22c', '23c', '70c', '80c', '12c', '32c', '04c', 'txt'],
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

  // Drag and drop handlers
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

  // Browser file input handler
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  }, [handleFiles]);

  // Open file dialog
  const openFileDialog = useCallback(() => {
    if (isTauri) {
      handleTauriFileSelect();
    } else {
      fileInputRef.current?.click();
    }
  }, [handleTauriFileSelect]);

  // Get icon and color for file type
  const getFileIcon = (file: WorkspaceFile) => {
    if (file.status === 'error') {
      return <ErrorIcon color="error" />;
    }
    if (file.status === 'parsing' || file.status === 'pending') {
      return <HourglassEmpty color="action" />;
    }
    if (file.type === 'ace') {
      return <Science color="primary" />;
    }
    if (file.type === 'endf') {
      return <InsertDriveFile color="secondary" />;
    }
    return <InsertDriveFile />;
  };

  return (
    <Drawer
      anchor="left"
      open={open}
      onClose={onClose}
      variant="persistent"
      sx={{
        width: open ? width : 0,
        flexShrink: 0,
        transition: 'width 0.3s ease-in-out',
        '& .MuiDrawer-paper': {
          width: width,
          boxSizing: 'border-box',
          height: '100%',
          position: 'relative',
          border: 'none',
          top: 'auto',
        },
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header */}
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6" component="div">
            📁 File Workspace
          </Typography>
          <IconButton onClick={onClose} size="small">
            <Close />
          </IconButton>
        </Box>

        <Divider />

        {/* Stats */}
        <Box sx={{ px: 2, py: 1, display: 'flex', gap: 1 }}>
          <Badge badgeContent={aceCount} color="primary">
            <Chip label="ACE" size="small" icon={<Science />} />
          </Badge>
          <Badge badgeContent={endfCount} color="secondary">
            <Chip label="ENDF" size="small" icon={<InsertDriveFile />} />
          </Badge>
        </Box>

        <Divider />

        {/* Upload Area */}
        <Box
          sx={{
            p: 2,
            bgcolor: dragActive ? 'action.hover' : 'background.default',
            border: dragActive ? '2px dashed' : '2px dashed transparent',
            borderColor: 'primary.main',
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
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
          />
          
          <Button
            variant="contained"
            startIcon={uploading ? <CircularProgress size={20} /> : <UploadFile />}
            onClick={openFileDialog}
            disabled={uploading}
            fullWidth
          >
            {uploading ? 'Uploading...' : 'Upload Files'}
          </Button>

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, textAlign: 'center' }}>
            Drop files here or click to browse
          </Typography>
        </Box>

        <Divider />

        {/* Filters */}
        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <TextField
            size="small"
            placeholder="Search files..."
            value={filter.searchQuery}
            onChange={(e) => setFilter({ searchQuery: e.target.value })}
            InputProps={{
              startAdornment: <Search sx={{ mr: 1, color: 'action.active' }} />,
            }}
          />

          <Box sx={{ display: 'flex', gap: 1 }}>
            <FormControl size="small" fullWidth>
              <InputLabel>Type</InputLabel>
              <Select
                value={filter.type}
                label="Type"
                onChange={(e) => setFilter({ type: e.target.value as FileType | 'all' })}
                startAdornment={<FilterList sx={{ mr: 1, color: 'action.active' }} />}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="ace">ACE</MenuItem>
                <MenuItem value="endf">ENDF</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" fullWidth>
              <InputLabel>Sort</InputLabel>
              <Select
                value={filter.sortBy}
                label="Sort"
                onChange={(e) => setFilter({ sortBy: e.target.value as any })}
                startAdornment={<Sort sx={{ mr: 1, color: 'action.active' }} />}
              >
                <MenuItem value="date">Date</MenuItem>
                <MenuItem value="name">Name</MenuItem>
                <MenuItem value="type">Type</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>

        <Divider />

        {/* File List */}
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          {filteredFiles.length === 0 ? (
            <Alert severity="info" sx={{ m: 2 }}>
              {files.length === 0 
                ? 'No files uploaded yet. Upload ACE or ENDF files to get started!'
                : 'No files match the current filters.'}
            </Alert>
          ) : (
            <List dense>
              {filteredFiles.map((file) => (
                <ListItem
                  key={file.id}
                  secondaryAction={
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={() => removeFile(file.id)}
                      title="Remove file"
                    >
                      <Delete />
                    </IconButton>
                  }
                  sx={{
                    borderLeft: file.status === 'error' ? '3px solid' : 'none',
                    borderColor: 'error.main',
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    {getFileIcon(file)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" noWrap>
                          {file.name}
                        </Typography>
                        {file.status === 'ready' && (
                          <CheckCircle color="success" sx={{ fontSize: 16 }} />
                        )}
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="caption" component="span" display="block">
                          {file.type?.toUpperCase() || 'Unknown'} • {formatFileSize(file.size)}
                        </Typography>
                        {file.metadata && 'zaid' in file.metadata && (
                          <Typography variant="caption" component="span" display="block" color="text.secondary">
                            ZAID: {file.metadata.zaid}
                          </Typography>
                        )}
                        {isENDFMetadata(file.metadata) && file.metadata.isotope && (
                          <Typography variant="caption" component="span" display="block" color="text.secondary">
                            Isotope: {file.metadata.isotope}
                          </Typography>
                        )}
                        {isENDFMetadata(file.metadata) && (
                          <Typography variant="caption" component="span" display="block" color="text.secondary">
                            MF4 MTs: {file.metadata.angular_mts.length} • MF34 MTs: {file.metadata.uncertainty_mts.length}
                          </Typography>
                        )}
                        {file.error && (
                          <Typography variant="caption" component="span" display="block" color="error">
                            {file.error}
                          </Typography>
                        )}
                      </>
                    }
                  />
                </ListItem>
              ))}
            </List>
          )}
        </Box>

        {/* Footer Actions */}
        {files.length > 0 && (
          <>
            <Divider />
            <Box sx={{ p: 2 }}>
              <Button
                variant="outlined"
                color="error"
                startIcon={<DeleteSweep />}
                onClick={clearAll}
                fullWidth
                size="small"
              >
                Clear All Files
              </Button>
            </Box>
          </>
        )}
      </Box>
    </Drawer>
  );
};
