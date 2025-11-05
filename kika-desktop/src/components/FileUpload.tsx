import React, { useState, useRef } from 'react';
import { Box, Button, Typography, Alert, CircularProgress, Chip } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

// Check if running in Tauri
const isTauri = '__TAURI__' in window;

interface FileUploadProps {
  accept?: string[];
  onFileSelect: (file: { name: string; path: string; content: string }) => void;
  label?: string;
  maxSizeMB?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  accept = [],
  onFileSelect,
  label = 'Select File',
  maxSizeMB = 100,
}) => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handler for Tauri native file dialog
  const handleTauriFileSelect = async () => {
    setError(null);
    setLoading(true);

    try {
      const { open } = await import('@tauri-apps/api/dialog');
      const { readTextFile } = await import('@tauri-apps/api/fs');

      const selected = await open({
        multiple: false,
        filters: accept.length > 0 ? [
          {
            name: 'Supported Files',
            extensions: accept,
          },
        ] : undefined,
      });

      if (!selected || Array.isArray(selected)) {
        setLoading(false);
        return;
      }

      const filePath = selected as string;
      const fileName = filePath.split(/[\\/]/).pop() || 'unknown';

      // Read file content
      const content = await readTextFile(filePath);

      // Check file size
      const sizeInMB = new Blob([content]).size / (1024 * 1024);
      if (sizeInMB > maxSizeMB) {
        setError(`File is too large (${sizeInMB.toFixed(1)}MB). Maximum size is ${maxSizeMB}MB.`);
        setLoading(false);
        return;
      }

      setSelectedFile(fileName);
      onFileSelect({ name: fileName, path: filePath, content });
    } catch (err) {
      console.error('Error selecting file:', err);
      setError('Failed to read file. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Handler for browser file input
  const handleBrowserFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setLoading(true);

    try {
      // Check file size
      const sizeInMB = file.size / (1024 * 1024);
      if (sizeInMB > maxSizeMB) {
        setError(`File is too large (${sizeInMB.toFixed(1)}MB). Maximum size is ${maxSizeMB}MB.`);
        setLoading(false);
        return;
      }

      // Read file content
      const content = await file.text();

      setSelectedFile(file.name);
      onFileSelect({ name: file.name, path: file.name, content });
    } catch (err) {
      console.error('Error reading file:', err);
      setError('Failed to read file. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = () => {
    if (isTauri) {
      handleTauriFileSelect();
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <Box sx={{ my: 2 }}>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {label}
      </Typography>
      
      {/* Hidden file input for browser mode */}
      {!isTauri && (
        <input
          ref={fileInputRef}
          type="file"
          accept={accept.map(ext => `.${ext}`).join(',')}
          onChange={handleBrowserFileSelect}
          style={{ display: 'none' }}
        />
      )}
      
      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
        <Button
          variant="outlined"
          startIcon={loading ? <CircularProgress size={20} /> : <UploadFileIcon />}
          onClick={handleFileSelect}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Browse Files'}
        </Button>

        {selectedFile && (
          <Chip
            icon={<CheckCircleIcon />}
            label={selectedFile}
            color="success"
            variant="outlined"
          />
        )}
      </Box>

      {accept.length > 0 && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Accepted formats: {accept.map(ext => `.${ext}`).join(', ')}
        </Typography>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      {!isTauri && (
        <Alert severity="info" sx={{ mt: 2 }}>
          💡 Running in browser mode. In the desktop app, you'll have native file dialogs.
        </Alert>
      )}
    </Box>
  );
};
