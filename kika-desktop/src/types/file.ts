/**
 * File types supported by the application
 */
export type FileType = 'ace' | 'endf';

/**
 * Status of file detection/parsing
 */
export type FileStatus = 'pending' | 'parsing' | 'ready' | 'error';

/**
 * Represents a file in the workspace
 */
export interface WorkspaceFile {
  id: string;
  name: string;
  path: string;
  content: string;
  type: FileType | null;
  status: FileStatus;
  size: number;
  uploadedAt: Date;
  error?: string;
  metadata?: ACEMetadata | ENDFMetadata;
}

/**
 * ACE file metadata from parsing
 */
export interface ACEMetadata {
  file_id: string;
  zaid: string;
  atomic_weight_ratio: number;
  temperature: number;
  available_reactions: number[];
  angular_reactions: number[];
  has_angular_distributions: boolean;
  energy_grid_size: number;
}

/**
 * ENDF file metadata from parsing
 */
export interface ENDFMetadata {
  file_id: string; // Required - always returned by backend
  zaid: string | null; // ZAID identifier (can be null if not determinable)
  isotope: string | null; // Human-readable isotope name (e.g., "U-235")
  mat: number | null; // Material number from ENDF header
  has_mf4: boolean; // Whether file contains MF4 (angular distributions)
  has_mf34: boolean; // Whether file contains MF34 (uncertainties)
  angular_mts: number[]; // Available MT numbers for angular distributions (MF4)
  uncertainty_mts: number[]; // Available MT numbers for uncertainties (MF34)
  max_legendre_order_by_mt: Record<string, number>; // Max Legendre order per MT
  // New: explicit available orders per MT for MF4 and MF34 (optional for backward-compat)
  available_orders_mf4_by_mt?: Record<string, number[]>;
  available_orders_mf34_by_mt?: Record<string, number[]>;
}

/**
 * File filter options
 */
export interface FileFilter {
  type: FileType | 'all';
  searchQuery: string;
  sortBy: 'name' | 'date' | 'type';
  sortOrder: 'asc' | 'desc';
}
