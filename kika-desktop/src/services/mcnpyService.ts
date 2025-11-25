/**
 * MCNPy Processing Service
 * 
 * Client for interacting with the local MCNPy Python server
 */

import { MCNPY_SERVER_URL } from '../config';
import type { ENDFMetadata } from '../types/file';

export interface ACEInfo {
  file_id: string;
  zaid: string;
  atomic_weight_ratio: number;
  temperature: number;
  available_reactions: number[];
  angular_reactions: number[];
  has_angular_distributions: boolean;
  energy_grid_size: number;
}

export type ENDFInfo = ENDFMetadata;

export type ENDFDataType = 'angular' | 'uncertainty';

export interface ENDFSeriesRequest {
  file_id?: string;
  file_content?: string;
  file_name: string;
  data_type: ENDFDataType;
  mt_number: number;
  order: number;
  include_uncertainty?: boolean;
  sigma?: number;
}

export interface ENDFUncertainty {
  lower: number[];
  upper: number[];
  sigma: number;
  label?: string;
  kind: 'absolute' | string;
}

export interface ENDFSeriesResponse {
  series_id: string;
  label: string;
  x: number[];
  y: number[];
  x_unit: string;
  y_unit: string;
  plot_style: string;
  line_shape?: string;
  metadata: Record<string, unknown>;
  suggested: Record<string, unknown>;
  uncertainty?: ENDFUncertainty;
}

export interface PlotImage {
  image_base64: string;
  format: string;
}

export interface PlotRequest {
  file_content?: string;
  file_name: string;
  plot_type: 'xs' | 'angular';
  mt_number?: number;
  energy?: number;  // Energy for angular distributions (MeV)
  energy_min?: number;
  energy_max?: number;
  file_id?: string;
}

export interface SeriesRequest {
  file_id?: string;
  file_content?: string;
  file_name: string;
  plot_type: 'xs' | 'angular';
  mt_number: number;
  energy?: number;
}

export interface SeriesDataResponse {
  series_id: string;
  label: string;
  x: number[];
  y: number[];
  x_unit: string;
  y_unit: string;
  metadata: Record<string, unknown>;
  suggested: Record<string, unknown>;
}

/**
 * Check if MCNPy server is running
 */
export async function checkMCNPyHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${MCNPY_SERVER_URL}/healthz`);
    return response.ok;
  } catch (error) {
    console.error('MCNPy server health check failed:', error);
    return false;
  }
}

/**
 * Parse ACE file and get basic information
 */
export async function parseACEFile(fileContent: string, fileName: string): Promise<ACEInfo> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/ace/parse`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_content: fileContent,
      file_name: fileName,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to parse ACE file');
  }

  return response.json();
}

/**
 * Parse ENDF file and get metadata
 */
export async function parseENDFFile(fileContent: string, fileName: string): Promise<ENDFInfo> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/endf/parse`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      file_content: fileContent,
      file_name: fileName,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to parse ENDF file');
  }

  return response.json();
}

/**
 * Generate plot from ACE file data
 */
export async function generateACEPlot(request: PlotRequest): Promise<PlotImage> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/ace/plot`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate plot');
  }

  return response.json();
}

/**
 * Fetch raw x/y data for a plotting series
 */
export async function fetchSeriesData(request: SeriesRequest): Promise<SeriesDataResponse> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/ace/series`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to load series data');
  }

  return response.json();
}

/**
 * Fetch ENDF series data for interactive plotting
 */
export async function fetchENDFSeries(request: ENDFSeriesRequest): Promise<ENDFSeriesResponse> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/endf/series`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to load ENDF data');
  }

  return response.json();
}

/**
 * Upload ACE file directly (alternative endpoint using multipart/form-data)
 */
export async function uploadACEFile(file: File): Promise<ACEInfo> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${MCNPY_SERVER_URL}/api/ace/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to upload ACE file');
  }

  return response.json();
}

/**
 * Export request for Matplotlib high-quality plot
 */
export interface MatplotlibExportRequest {
  series: Array<{
    fileName: string;
    fileId?: string;
    mtNumber?: number;
    energy?: number;
    plotType: string;
    color: string;
    lineWidth: number;
    lineStyle: string;
    showMarkers: boolean;
    markerSymbol: string;
    markerSize: number;
    labelMode: string;
    customLabel: string;
  }>;
  figure_settings: {
    title: string;
    xLabel: string;
    yLabel: string;
    logX: boolean;
    logY: boolean;
    xMin: string;
    xMax: string;
    yMin: string;
    yMax: string;
    showGrid: boolean;
    gridAlpha?: number;
    showMinorGrid?: boolean;
    minorGridAlpha?: number;
    showLegend: boolean;
    legendPosition: string;
    figWidthInches?: number;
    figHeightInches?: number;
    titleFontSize?: number;
    labelFontSize?: number;
    legendFontSize?: number;
    tickFontSizeX?: number;
    tickFontSizeY?: number;
    maxTicksX?: number;
    maxTicksY?: number;
    rotateTicksX?: number;
    rotateTicksY?: number;
    // Legacy fields for backward compatibility
    width?: number;
    height?: number;
  };
  style: string;
  export_format: string;
  dpi: number;
}

/**
 * Response from Matplotlib export
 */
export interface MatplotlibExportResponse {
  image_base64: string;
  format: string;
  width_px: number;
  height_px: number;
  dpi: number;
}

export interface ENDFMatplotlibExportRequest {
  series: Array<{
    file_name: string;
    file_id?: string;
    file_content?: string;
    data_type: ENDFDataType;
    mt_number: number;
    order: number;
    include_uncertainty?: boolean;
    sigma?: number;
    color: string;
    lineWidth: number;
    lineStyle: string;
    showMarkers: boolean;
    markerSymbol: string;
    markerSize: number;
    labelMode: string;
    customLabel: string;
  }>;
  figure_settings: MatplotlibExportRequest['figure_settings'];
  style: string;
  export_format: string;
  dpi: number;
}

/**
 * Export plot using Matplotlib for publication-quality output
 */
export async function exportWithMatplotlib(
  request: MatplotlibExportRequest
): Promise<MatplotlibExportResponse> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/plot/matplotlib-export`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to export with Matplotlib');
  }

  return response.json();
}

/**
 * Export ENDF plot using Matplotlib
 */
export async function exportENDFWithMatplotlib(
  request: ENDFMatplotlibExportRequest
): Promise<MatplotlibExportResponse> {
  const response = await fetch(`${MCNPY_SERVER_URL}/api/plot/endf/matplotlib-export`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to export ENDF plot');
  }

  return response.json();
}
