import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { WorkspaceFile, FileFilter, FileType } from '../types/file';
import { detectFileType, generateFileId } from '../utils/fileDetection';
import { parseACEFile, parseENDFFile } from '../services/mcnpyService';

interface FileWorkspaceContextType {
  files: WorkspaceFile[];
  filter: FileFilter;
  addFiles: (files: Array<{ name: string; path: string; content: string }>) => Promise<void>;
  removeFile: (id: string) => void;
  clearAll: () => void;
  setFilter: (filter: Partial<FileFilter>) => void;
  getFilteredFiles: () => WorkspaceFile[];
  getFilesByType: (type: FileType) => WorkspaceFile[];
}

const FileWorkspaceContext = createContext<FileWorkspaceContextType | undefined>(undefined);

const DEFAULT_FILTER: FileFilter = {
  type: 'all',
  searchQuery: '',
  sortBy: 'date',
  sortOrder: 'desc',
};

export const FileWorkspaceProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [filter, setFilterState] = useState<FileFilter>(DEFAULT_FILTER);

  // DO NOT persist files in localStorage
  // Files are session-only to avoid issues with:
  // 1. localStorage quota limits
  // 2. Stale file_id references (backend cache expires)
  // 3. Content-less file metadata causing errors
  // 4. Guest users having persistent state across sessions
  
  // Clean up any legacy localStorage data on mount
  useEffect(() => {
    const legacyData = localStorage.getItem('workspaceFiles');
    if (legacyData) {
      console.log('Removing legacy workspaceFiles from localStorage');
      localStorage.removeItem('workspaceFiles');
    }
  }, []);

  const parseFile = useCallback(async (file: WorkspaceFile): Promise<WorkspaceFile> => {
    try {
      if (file.type === 'ace') {
        setFiles(prev => prev.map(f => 
          f.id === file.id ? { ...f, status: 'parsing' as const } : f
        ));

        const metadata = await parseACEFile(file.content, file.name);
        
        return {
          ...file,
          status: 'ready',
          metadata,
        };
      } else if (file.type === 'endf') {
        setFiles(prev => prev.map(f =>
          f.id === file.id ? { ...f, status: 'parsing' as const } : f
        ));

        const metadata = await parseENDFFile(file.content, file.name);

        return {
          ...file,
          status: 'ready',
          metadata,
          error: undefined,
        };
      }
      
      return {
        ...file,
        status: 'ready',
      };
    } catch (error) {
      return {
        ...file,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }, []);

  const addFiles = useCallback(async (newFiles: Array<{ name: string; path: string; content: string }>) => {
    const workspaceFiles: WorkspaceFile[] = newFiles.map(file => {
      const detectedType = detectFileType(file.name, file.content);
      
      return {
        id: generateFileId(),
        name: file.name,
        path: file.path,
        content: file.content,
        type: detectedType,
        status: detectedType ? 'pending' : 'error',
        size: new Blob([file.content]).size,
        uploadedAt: new Date(),
        error: detectedType ? undefined : 'Could not detect file type. Expected ACE or ENDF-6 format.',
      };
    });

    setFiles(prev => [...prev, ...workspaceFiles]);

    // Parse files asynchronously
    for (const file of workspaceFiles) {
      if (file.type && file.status === 'pending') {
        const parsedFile = await parseFile(file);
        setFiles(prev => prev.map(f => f.id === file.id ? parsedFile : f));
      }
    }
  }, [parseFile]);

  const removeFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  }, []);

  const clearAll = useCallback(() => {
    setFiles([]);
    localStorage.removeItem('workspaceFiles');
  }, []);

  const setFilter = useCallback((newFilter: Partial<FileFilter>) => {
    setFilterState(prev => ({ ...prev, ...newFilter }));
  }, []);

  const getFilteredFiles = useCallback(() => {
    let filtered = [...files];

    // Apply type filter
    if (filter.type !== 'all') {
      filtered = filtered.filter(f => f.type === filter.type);
    }

    // Apply search filter
    if (filter.searchQuery) {
      const query = filter.searchQuery.toLowerCase();
      filtered = filtered.filter(f => 
        f.name.toLowerCase().includes(query) ||
        (f.metadata && 'zaid' in f.metadata && f.metadata.zaid?.toLowerCase().includes(query))
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (filter.sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'date':
          comparison = a.uploadedAt.getTime() - b.uploadedAt.getTime();
          break;
        case 'type':
          comparison = (a.type || '').localeCompare(b.type || '');
          break;
      }

      return filter.sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [files, filter]);

  const getFilesByType = useCallback((type: FileType) => {
    return files.filter(f => f.type === type && f.status === 'ready');
  }, [files]);

  const value: FileWorkspaceContextType = {
    files,
    filter,
    addFiles,
    removeFile,
    clearAll,
    setFilter,
    getFilteredFiles,
    getFilesByType,
  };

  return (
    <FileWorkspaceContext.Provider value={value}>
      {children}
    </FileWorkspaceContext.Provider>
  );
};

export const useFileWorkspace = () => {
  const context = useContext(FileWorkspaceContext);
  if (!context) {
    throw new Error('useFileWorkspace must be used within FileWorkspaceProvider');
  }
  return context;
};
