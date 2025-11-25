import React, { useEffect, useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import Plotly from 'plotly.js-dist-min';
import type {
  Config as PlotlyConfig,
  Layout as PlotlyLayout,
  PlotData,
  PlotlyHTMLElement,
} from 'plotly.js';
type PlotlyMarkerSymbol = NonNullable<PlotData['marker']>['symbol'];
type PlotTrace = Partial<PlotData>;
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  IconButton,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import RestoreOutlinedIcon from '@mui/icons-material/RestoreOutlined';
import SaveOutlinedIcon from '@mui/icons-material/SaveOutlined';
import HighQualityIcon from '@mui/icons-material/HighQuality';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import {
  fetchENDFSeries,
  exportENDFWithMatplotlib,
  type ENDFSeriesResponse,
} from '../services/mcnpyService';
import type { ENDFMetadata } from '../types/file';

const COLOR_PALETTE = [
  '#0173B2',
  '#DE8F05',
  '#029E73',
  '#D55E00',
  '#CC78BC',
  '#CA9161',
  '#FBAFE4',
  '#949494',
  '#ECE133',
  '#56B4E9',
];

const LINE_STYLE_OPTIONS: { label: string; value: SeriesConfig['lineStyle'] }[] = [
  { label: 'Solid', value: 'solid' },
  { label: 'Dashed', value: 'dash' },
  { label: 'Dotted', value: 'dot' },
  { label: 'Dash-dot', value: 'dashdot' },
];

const MARKER_OPTIONS: { label: string; value: string }[] = [
  { label: 'Circle', value: 'circle' },
  { label: 'Square', value: 'square' },
  { label: 'Triangle up', value: 'triangle-up' },
  { label: 'Triangle down', value: 'triangle-down' },
  { label: 'Diamond', value: 'diamond' },
  { label: 'Star', value: 'star' },
  { label: 'X', value: 'x' },
  { label: 'Plus', value: 'cross' },
];

const DATA_MODE_OPTIONS = [
  { label: 'Angular Distributions (MF4)', value: 'angular' },
  { label: 'Uncertainties (MF34)', value: 'uncertainty' },
] as const;

const LOCAL_STORAGE_KEY = 'kikaEndfViewerConfigs';
const DEFAULT_SIGMA = 1.0;
const MAX_ORDER_FALLBACK = 12;

type DataMode = (typeof DATA_MODE_OPTIONS)[number]['value'];

type NotificationState = { type: 'success' | 'error'; message: string } | null;

interface FigureSettings {
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
  gridAlpha: number;
  showMinorGrid: boolean;
  minorGridAlpha: number;
  showLegend: boolean;
  legendPosition: 'outside' | 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  figWidthInches: number;
  figHeightInches: number;
  titleFontSize: number;
  labelFontSize: number;
  legendFontSize: number;
  tickFontSizeX: number;
  tickFontSizeY: number;
  maxTicksX: number;
  maxTicksY: number;
  rotateTicksX: number;
  rotateTicksY: number;
}

interface SeriesConfig {
  id: string;
  fileName: string;
  fileId?: string;
  mtNumber?: number;
  order: number;
  includeUncertainty: boolean;
  sigma: number;
  labelMode: 'auto' | 'custom';
  customLabel: string;
  autoLabel?: string;
  color: string;
  lineWidth: number;
  lineStyle: 'solid' | 'dash' | 'dot' | 'dashdot';
  showMarkers: boolean;
  markerSymbol: PlotlyMarkerSymbol;
  markerSize: number;
}

interface SeriesDataState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  signature?: string;
  data?: ENDFSeriesResponse;
  error?: string;
}

interface SavedPlotConfig {
  id: string;
  name: string;
  createdAt: string;
  dataMode: DataMode;
  figure: FigureSettings;
  series: SeriesConfig[];
}

export interface LoadedENDFFile {
  id: string;
  name: string;
  path: string;
  content: string;
  metadata?: ENDFMetadata;
}

const createSeriesId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const getDefaultFigureSettings = (): FigureSettings => ({
  title: 'ENDF Angular Distribution',
  xLabel: 'Energy (eV)',
  yLabel: 'Legendre Coefficient',
  logX: true,
  logY: false,
  xMin: '',
  xMax: '',
  yMin: '',
  yMax: '',
  showGrid: true,
  gridAlpha: 0.3,
  showMinorGrid: false,
  minorGridAlpha: 0.15,
  showLegend: true,
  legendPosition: 'top-right',
  figWidthInches: 8,
  figHeightInches: 5,
  titleFontSize: 16,
  labelFontSize: 14,
  legendFontSize: 11,
  tickFontSizeX: 12,
  tickFontSizeY: 12,
  maxTicksX: 10,
  maxTicksY: 10,
  rotateTicksX: 0,
  rotateTicksY: 0,
});

const getAvailableMts = (file: LoadedENDFFile | undefined, mode: DataMode): number[] => {
  if (!file?.metadata) return [];
  return mode === 'angular'
    ? file.metadata.angular_mts || []
    : file.metadata.uncertainty_mts || [];
};

const getMaxOrder = (file: LoadedENDFFile | undefined, mt?: number): number => {
  if (!file?.metadata || mt === undefined) return MAX_ORDER_FALLBACK;
  const value = file.metadata.max_legendre_order_by_mt?.[String(mt)];
  return typeof value === 'number' && value > 0 ? value : MAX_ORDER_FALLBACK;
};

// Prefer explicit available order lists from backend; fallback to 0..max
const getAvailableOrders = (
  file: LoadedENDFFile | undefined,
  mode: DataMode,
  mt?: number
): number[] => {
  if (!file?.metadata || mt === undefined) return [];
  const key = String(mt);
  const meta = file.metadata as ENDFMetadata;
  if (mode === 'angular' && (meta as any).available_orders_mf4_by_mt?.[key]) {
    const list = (meta as any).available_orders_mf4_by_mt?.[key] as number[];
    if (Array.isArray(list) && list.length) return list;
  }
  if (mode === 'uncertainty' && (meta as any).available_orders_mf34_by_mt?.[key]) {
    const list = (meta as any).available_orders_mf34_by_mt?.[key] as number[];
    if (Array.isArray(list) && list.length) return list;
  }
  const max = getMaxOrder(file, mt);
  return Array.from({ length: max + 1 }, (_, i) => i);
};

const hexToRgba = (hex: string, alpha: number) => {
  const clean = hex.replace('#', '');
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const parseRange = (minStr: string, maxStr: string, isLog: boolean): [number, number] | undefined => {
  if (!minStr && !maxStr) return undefined;
  const min = minStr ? Number(minStr) : undefined;
  const max = maxStr ? Number(maxStr) : undefined;
  if ((minStr && Number.isNaN(min)) || (maxStr && Number.isNaN(max))) {
    return undefined;
  }
  if (min !== undefined && max !== undefined) {
    if (isLog && (min <= 0 || max <= 0)) return undefined;
    return isLog ? [Math.log10(min), Math.log10(max)] : [min, max];
  }
  if (min !== undefined) {
    if (isLog && min <= 0) return undefined;
    const maxVal = max ?? min * 10;
    return isLog ? [Math.log10(min), Math.log10(maxVal)] : [min, maxVal];
  }
  if (max !== undefined) {
    if (isLog && max <= 0) return undefined;
    const minVal = min ?? max / 10;
    return isLog ? [Math.log10(minVal), Math.log10(max)] : [minVal, max];
  }
  return undefined;
};

const legendPositionMap: Record<
  FigureSettings['legendPosition'],
  { x?: number; y?: number; xanchor?: 'left' | 'center' | 'right'; yanchor?: 'top' | 'middle' | 'bottom' }
> = {
  outside: { x: 1.02, y: 1, xanchor: 'left', yanchor: 'top' },
  'top-right': { x: 1, y: 1, xanchor: 'right', yanchor: 'top' },
  'top-left': { x: 0, y: 1, xanchor: 'left', yanchor: 'top' },
  'bottom-right': { x: 1, y: 0, xanchor: 'right', yanchor: 'bottom' },
  'bottom-left': { x: 0, y: 0, xanchor: 'left', yanchor: 'bottom' },
};

interface ENDFPlotViewerProps {
  files: LoadedENDFFile[];
}

const formatTimestamp = (iso: string) => {
  const date = new Date(iso);
  return date.toLocaleString();
};

export const ENDFPlotViewer: React.FC<ENDFPlotViewerProps> = ({ files }) => {
  const [dataMode, setDataMode] = useState<DataMode>('angular');
  const [figureSettings, setFigureSettings] = useState<FigureSettings>(() => getDefaultFigureSettings());
  const [seriesConfigs, setSeriesConfigs] = useState<SeriesConfig[]>([]);
  const [seriesDataMap, setSeriesDataMap] = useState<Record<string, SeriesDataState>>({});
  const [notification, setNotification] = useState<NotificationState>(null);
  const [configName, setConfigName] = useState('');
  // Do NOT persist saved configs - they should be session-only
  const [savedConfigs, setSavedConfigs] = useState<SavedPlotConfig[]>([]);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [matplotlibExportSettings, setMatplotlibExportSettings] = useState({ style: 'publication', format: 'png', dpi: 300 });
  const [previewImage, setPreviewImage] = useState<{
    base64: string;
    format: string;
    width: number;
    height: number;
    dpi: number;
  } | null>(null);
  const [generatingPreview, setGeneratingPreview] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportOptions] = useState({ format: 'png', width: 1200, height: 720, scale: 2 });
  const plotRef = useRef<PlotlyHTMLElement | null>(null);

  // Clean up any legacy localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const legacyData = window.localStorage.getItem(LOCAL_STORAGE_KEY);
    if (legacyData) {
      console.log(`Removing legacy ${LOCAL_STORAGE_KEY} from localStorage`);
      window.localStorage.removeItem(LOCAL_STORAGE_KEY);
    }
  }, []);
  
  // Do NOT persist saved configs to localStorage - they are session-only

  useEffect(() => {
    // Update figure title/labels based on mode but keep other settings
    setFigureSettings(prev => ({
      ...prev,
      title: dataMode === 'angular' ? 'Angular Distribution Comparison' : 'Angular Distribution Uncertainty',
      yLabel: dataMode === 'angular' ? 'Legendre Coefficient' : 'Relative Uncertainty',
    }));
    setSeriesConfigs([]);
    setSeriesDataMap({});
  }, [dataMode]);

  useEffect(() => {
    setSeriesConfigs((prev) =>
      prev.map((series) => {
        const file = files.find((f) => f.name === series.fileName);
        if (file?.metadata?.file_id && series.fileId !== file.metadata.file_id) {
          return { ...series, fileId: file.metadata.file_id };
        }
        return series;
      })
    );
  }, [files]);

  useEffect(() => {
    seriesConfigs.forEach((series) => {
      const file = files.find((f) => f.name === series.fileName);
      if (!file || series.mtNumber === undefined) {
        return;
      }
      
      // If file has no content, it might be a stale entry - skip it
      if (!file.content || file.content.trim() === '') {
        setSeriesDataMap((prev) => ({
          ...prev,
          [series.id]: { 
            status: 'error', 
            signature: '', 
            error: 'File content not available. Please re-upload the file.' 
          },
        }));
        return;
      }
      
      // Validate that the selected order is available for this MT and mode
      const availableOrdersForMt = getAvailableOrders(file, dataMode, series.mtNumber);
      if (availableOrdersForMt.length && !availableOrdersForMt.includes(series.order)) {
        const nextOrder = availableOrdersForMt.includes(1) ? 1 : availableOrdersForMt[0];
        handleUpdateSeries(series.id, { order: nextOrder });
        return;
      }
      
      // Always use file_id if available, otherwise use content
      // Note: file_id might be stale if backend cache expired, so we include content as fallback
      const fileId = file.metadata?.file_id;
      const signature = [
        dataMode,
        fileId || file.content.slice(0, 100), // Use content hash if no file_id
        series.mtNumber,
        series.order,
        series.includeUncertainty,
        series.sigma,
      ].join('|');
      
      const existing = seriesDataMap[series.id];
      if (existing && existing.signature === signature && existing.status === 'ready') {
        return;
      }
      if (existing && existing.status === 'loading' && existing.signature === signature) {
        return;
      }
      setSeriesDataMap((prev) => ({
        ...prev,
        [series.id]: { status: 'loading', signature },
      }));
      
      // ALWAYS send file_content as fallback in case file_id is stale
      const requestPayload = {
        file_id: fileId,
        file_content: file.content, // Always include content as backup
        file_name: file.name,
        data_type: dataMode,
        mt_number: series.mtNumber,
        order: series.order,
        include_uncertainty: dataMode === 'angular' ? series.includeUncertainty : false,
        sigma: series.sigma,
      };
      
      console.log('[ENDFPlotViewer] Fetching series data with:', {
        fileName: file.name,
        dataType: dataMode,
        mtNumber: series.mtNumber,
        order: series.order,
        includeUncertainty: requestPayload.include_uncertainty,
        sigma: series.sigma,
      });
      
      fetchENDFSeries(requestPayload)
        .then((data) => {
          console.log('[ENDFPlotViewer] 📥 RAW Response from /api/endf/series:', JSON.stringify(data, null, 2).substring(0, 500));
          console.log('[ENDFPlotViewer] Received series data:', {
            label: data.label,
            xLength: data.x?.length,
            yLength: data.y?.length,
            hasUncertainty: !!data.uncertainty,
            uncertaintyType: typeof data.uncertainty,
            uncertaintyKeys: data.uncertainty ? Object.keys(data.uncertainty) : [],
            rawUncertaintyValue: data.uncertainty,
          });
          
          // Detailed uncertainty inspection
          if (data.uncertainty) {
            console.log('[ENDFPlotViewer] 🔍 Detailed uncertainty data:', {
              hasLower: !!data.uncertainty.lower,
              hasUpper: !!data.uncertainty.upper,
              lowerLength: data.uncertainty.lower?.length || 0,
              upperLength: data.uncertainty.upper?.length,
              lowerSample: data.uncertainty.lower?.slice(0, 3),
              upperSample: data.uncertainty.upper?.slice(0, 3),
              sigma: data.uncertainty.sigma,
              kind: data.uncertainty.kind,
            });
          } else if (requestPayload.include_uncertainty) {
            console.warn('[ENDFPlotViewer] ⚠️ Requested uncertainty but backend returned null/undefined');
          }
          
          setSeriesDataMap((prev) => ({
            ...prev,
            [series.id]: { status: 'ready', signature, data },
          }));
          setSeriesConfigs((prev) =>
            prev.map((cfg) =>
              cfg.id === series.id
                ? { ...cfg, fileId: file.metadata?.file_id, autoLabel: data.label }
                : cfg
            )
          );
        })
        .catch((error) => {
          setSeriesDataMap((prev) => ({
            ...prev,
            [series.id]: {
              status: 'error',
              signature,
              error: error instanceof Error ? error.message : 'Failed to load data',
            },
          }));
        });
    });
  }, [seriesConfigs, files, dataMode, seriesDataMap]);

  const handleAddSeries = () => {
    if (files.length === 0) {
      setNotification({ 
        type: 'error', 
        message: 'Upload ENDF files first. Use the File Workspace (📁) to add files.' 
      });
      return;
    }
    const candidates = files.filter((file) => getAvailableMts(file, dataMode).length > 0);
    if (candidates.length === 0) {
      const modeDescription = dataMode === 'angular' ? 'MF4 (angular distributions)' : 'MF34 (uncertainties)';
      setNotification({ 
        type: 'error', 
        message: `None of the loaded files contain ${modeDescription} data. Please upload ENDF files with the required data sections.` 
      });
      return;
    }
    const defaultFile = candidates[0];
    const availableMts = getAvailableMts(defaultFile, dataMode);
    const defaultMt = availableMts[0];
    const availableOrders = getAvailableOrders(defaultFile, dataMode, defaultMt);
    const maxOrder = availableOrders.length ? Math.max(...availableOrders) : getMaxOrder(defaultFile, defaultMt);
    const canUseUncertainty =
      dataMode === 'angular' &&
      Boolean(defaultFile.metadata?.has_mf34 && defaultFile.metadata.uncertainty_mts.includes(defaultMt));
  const defaultOrder = availableOrders.includes(1)
    ? 1
    : (availableOrders.length ? availableOrders[0] : Math.max(0, Math.min(1, maxOrder)));

  const nextSeries: SeriesConfig = {
    id: createSeriesId(),
    fileName: defaultFile.name,
    fileId: defaultFile.metadata?.file_id,
    mtNumber: defaultMt,
    order: defaultOrder,
      includeUncertainty: canUseUncertainty,
      sigma: DEFAULT_SIGMA,
      labelMode: 'auto',
      customLabel: '',
      color: COLOR_PALETTE[seriesConfigs.length % COLOR_PALETTE.length],
      lineWidth: 2,
      lineStyle: 'solid',
      showMarkers: false,
      markerSymbol: 'circle',
      markerSize: 8,
    };
    setSeriesConfigs((prev) => [...prev, nextSeries]);
  };

  const handleUpdateSeries = (id: string, patch: Partial<SeriesConfig>) => {
    setSeriesConfigs((prev) => prev.map((series) => (series.id === id ? { ...series, ...patch } : series)));
  };

  const handleRemoveSeries = (id: string) => {
    setSeriesConfigs((prev) => prev.filter((series) => series.id !== id));
    setSeriesDataMap((prev) => {
      const clone = { ...prev };
      delete clone[id];
      return clone;
    });
  };

  const handleClearSeries = () => {
    setSeriesConfigs([]);
    setSeriesDataMap({});
  };

  const plotlyData: PlotTrace[] = useMemo(() => {
    const traces: PlotTrace[] = [];
    seriesConfigs.forEach((series, index) => {
      const state = seriesDataMap[series.id];
      if (!state || state.status !== 'ready' || !state.data) return;
      const label =
        series.labelMode === 'custom' && series.customLabel.trim().length > 0
          ? series.customLabel.trim()
          : series.autoLabel || state.data.label || `Series ${index + 1}`;

      console.log(`[ENDFPlotViewer] Processing series ${label}:`, {
        includeUncertainty: series.includeUncertainty,
        hasUncertaintyData: !!state.data.uncertainty,
        uncertaintyDetails: state.data.uncertainty ? {
          hasUpper: !!state.data.uncertainty.upper,
          hasLower: !!state.data.uncertainty.lower,
          upperLength: state.data.uncertainty.upper?.length,
          lowerLength: state.data.uncertainty.lower?.length,
          dataLength: state.data.x.length,
          sigma: state.data.uncertainty.sigma,
        } : 'no uncertainty data',
      });

      // Add uncertainty band if available and arrays match
      if (state.data.uncertainty) {
        const { upper, lower } = state.data.uncertainty;
        // Validate that uncertainty arrays match data length
        if (upper && lower && 
            upper.length === state.data.x.length && 
            lower.length === state.data.x.length) {
          console.log(`[ENDFPlotViewer] ✅ Adding uncertainty band for ${label}`);
          
          const fillColor = hexToRgba(series.color, 0.15);
          const bandXData = [...state.data.x, ...state.data.x.slice().reverse()];
          const bandYData = [...upper, ...lower.slice().reverse()];
          
          console.log(`[ENDFPlotViewer] 📊 Band trace details:`, {
            fillColor,
            xDataLength: bandXData.length,
            yDataLength: bandYData.length,
            xSample: bandXData.slice(0, 3),
            ySample: bandYData.slice(0, 3),
            upperSample: upper.slice(0, 3),
            lowerSample: lower.slice(0, 3),
            originalXSample: state.data.x.slice(0, 3),
            originalYSample: state.data.y.slice(0, 3),
          });
          
          const bandTrace: PlotTrace = {
            x: bandXData,
            y: bandYData,
            type: 'scatter',
            mode: 'lines',
            fill: 'toself',
            fillcolor: fillColor,
            line: { color: 'transparent' },
            hoverinfo: 'skip',
            showlegend: false,
            name: `${label} ±${state.data.uncertainty.sigma.toFixed(1)}σ`,
          };
          traces.push(bandTrace);
          console.log(`[ENDFPlotViewer] 📈 Band trace added. Total traces so far: ${traces.length}`);
        } else {
          console.warn(`[ENDFPlotViewer] ❌ Uncertainty data length mismatch for series ${label}:`, {
            dataLength: state.data.x.length,
            upperLength: upper?.length || 0,
            lowerLength: lower?.length || 0,
          });
        }
      } else if (series.includeUncertainty) {
        console.warn(`[ENDFPlotViewer] ⚠️ Series ${label} has includeUncertainty=true but no uncertainty data received from backend`);
      }

      const trace: PlotTrace = {
        x: state.data.x,
        y: state.data.y,
        name: label,
        type: 'scatter',
        mode: series.showMarkers ? 'lines+markers' : 'lines',
        line: {
          color: series.color,
          width: series.lineWidth,
          dash: series.lineStyle,
          shape: (state.data.line_shape as any) || (state.data.plot_style === 'step' ? 'hv' : 'linear'),
        },
      };

      if (series.showMarkers) {
        trace.marker = {
          color: series.color,
          size: series.markerSize,
          symbol: series.markerSymbol,
        };
      }

      traces.push(trace);
    });
    
    console.log(`[ENDFPlotViewer] 🎨 Final plotlyData summary:`, {
      totalTraces: traces.length,
      traceTypes: traces.map(t => t.type),
      traceModes: traces.map(t => t.mode),
      traceFills: traces.map(t => t.fill || 'none'),
      traceNames: traces.map(t => t.name),
    });
    
    return traces;
  }, [seriesConfigs, seriesDataMap]);

  const layout: Partial<PlotlyLayout> = useMemo(() => {
    const xRange = parseRange(figureSettings.xMin, figureSettings.xMax, figureSettings.logX);
    const yRange = parseRange(figureSettings.yMin, figureSettings.yMax, figureSettings.logY);
    const DPI = 96;
    const widthPx = Math.round(figureSettings.figWidthInches * DPI);
    const heightPx = Math.round(figureSettings.figHeightInches * DPI);
    const gridColor = `rgba(0,0,0,${figureSettings.gridAlpha})`;

    return {
      title: {
        text: figureSettings.title,
        font: {
          size: figureSettings.titleFontSize,
          family: 'Computer Modern, Times New Roman, serif',
          color: '#000000',
        },
      },
      paper_bgcolor: '#FFFFFF',
      plot_bgcolor: '#FFFFFF',
      width: widthPx,
      height: heightPx,
      margin: { t: 60, r: 40, b: 70, l: 80 },
      xaxis: {
        title: {
          text: figureSettings.xLabel,
          font: {
            size: figureSettings.labelFontSize,
            family: 'Computer Modern, Times New Roman, serif',
            color: '#000000',
          },
        },
        type: (figureSettings.logX ? 'log' : 'linear') as any,
        ...(xRange ? { range: xRange } : { autorange: true }),
        showgrid: figureSettings.showGrid,
        gridcolor: gridColor,
        gridwidth: 1,
        showline: true,
        linewidth: 1.5,
        linecolor: '#000000',
        mirror: true,
        ticks: 'outside' as const,
        tickwidth: 1.5,
        tickcolor: '#000000',
        ticklen: 6,
        tickangle: -figureSettings.rotateTicksX,
        tickfont: {
          size: figureSettings.tickFontSizeX,
          family: 'Computer Modern, Times New Roman, serif',
          color: '#000000',
        },
        nticks: figureSettings.maxTicksX,
        zeroline: false,
      },
      yaxis: {
        title: {
          text: figureSettings.yLabel,
          font: {
            size: figureSettings.labelFontSize,
            family: 'Computer Modern, Times New Roman, serif',
            color: '#000000',
          },
        },
        type: (figureSettings.logY ? 'log' : 'linear') as any,
        ...(yRange ? { range: yRange } : { autorange: true }),
        showgrid: figureSettings.showGrid,
        gridcolor: gridColor,
        gridwidth: 1,
        showline: true,
        linewidth: 1.5,
        linecolor: '#000000',
        mirror: true,
        ticks: 'outside' as const,
        tickwidth: 1.5,
        tickcolor: '#000000',
        ticklen: 6,
        tickangle: -figureSettings.rotateTicksY,
        tickfont: {
          size: figureSettings.tickFontSizeY,
          family: 'Computer Modern, Times New Roman, serif',
          color: '#000000',
        },
        nticks: figureSettings.maxTicksY,
        zeroline: false,
      },
      showlegend: figureSettings.showLegend,
      legend: figureSettings.showLegend
        ? {
            ...legendPositionMap[figureSettings.legendPosition],
            bgcolor: 'rgba(255,255,255,0.95)',
            bordercolor: '#000000',
            borderwidth: 1,
            font: {
              size: figureSettings.legendFontSize,
              family: 'Computer Modern, Times New Roman, serif',
              color: '#000000',
            },
          }
        : undefined,
      font: {
        family: 'Computer Modern, Times New Roman, serif',
        color: '#000000',
      },
    };
  }, [figureSettings]);

  const config: Partial<PlotlyConfig> = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  };

  const handleQuickExport = async () => {
    if (!plotRef.current || plotlyData.length === 0) {
      setNotification({ type: 'error', message: 'No plot data available to export' });
      return;
    }
    setExporting(true);
    try {
      const dataUrl = await Plotly.toImage(plotRef.current, {
        format: exportOptions.format as any,
        width: exportOptions.width,
        height: exportOptions.height,
        scale: exportOptions.scale,
      });
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = `endf_plot.${exportOptions.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setNotification({ type: 'success', message: 'Plot exported successfully' });
    } catch (error) {
      setNotification({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to export plot',
      });
    } finally {
      setExporting(false);
    }
  };

  const handleSaveConfig = () => {
    if (seriesConfigs.length === 0) {
      setNotification({ type: 'error', message: 'Add at least one series before saving' });
      return;
    }
    const name = configName.trim() || `ENDF Plot - ${new Date().toLocaleString()}`;
    const snapshot: SavedPlotConfig = {
      id: createSeriesId(),
      name,
      createdAt: new Date().toISOString(),
      dataMode,
      figure: { ...figureSettings },
      series: seriesConfigs.map((series) => ({ ...series })),
    };
    setSavedConfigs((prev) => {
      const existingIndex = prev.findIndex((cfg) => cfg.name === name);
      if (existingIndex >= 0) {
        const clone = [...prev];
        clone[existingIndex] = { ...snapshot, id: prev[existingIndex].id };
        return clone;
      }
      return [snapshot, ...prev];
    });
    setNotification({ type: 'success', message: `Saved configuration "${name}"` });
  };

  const handleRestoreConfig = (configId: string) => {
    const target = savedConfigs.find((cfg) => cfg.id === configId);
    if (!target) return;
    const availableSeries = target.series.filter((series) =>
      files.some((file) => file.name === series.fileName)
    );
    if (availableSeries.length === 0) {
      setNotification({ type: 'error', message: 'Required files are not loaded' });
      return;
    }
    setDataMode(target.dataMode);
    setFigureSettings({ ...target.figure });
    setSeriesConfigs(availableSeries.map((series) => ({ ...series, id: createSeriesId() })));
    setSeriesDataMap({});
    const missing = target.series.length - availableSeries.length;
    const message = missing > 0
      ? `Restored "${target.name}" (skipped ${missing} missing file${missing > 1 ? 's' : ''})`
      : `Restored "${target.name}"`;
    setNotification({ type: 'success', message });
  };

  const handleDeleteConfig = (configId: string) => {
    setSavedConfigs((prev) => prev.filter((cfg) => cfg.id !== configId));
  };

  const prepareExportData = () => ({
    series: seriesConfigs
      .filter((series) => series.mtNumber !== undefined)
      .map((series) => {
        // Get the file to include content as fallback
        const file = files.find(f => f.name === series.fileName);
        return {
          file_name: series.fileName,
          file_id: series.fileId,
          file_content: file?.content,  // Include content as backup
          data_type: dataMode,
          mt_number: series.mtNumber!,
          order: series.order,
          include_uncertainty: dataMode === 'angular' ? series.includeUncertainty : false,
          sigma: series.sigma,
          color: series.color,
          lineWidth: series.lineWidth,
          lineStyle: series.lineStyle,
          showMarkers: series.showMarkers,
          markerSymbol: String(series.markerSymbol || 'circle'),
          markerSize: series.markerSize,
          labelMode: series.labelMode,
          customLabel: series.customLabel,
        };
      }),
    figure_settings: {
      title: figureSettings.title,
      xLabel: figureSettings.xLabel,
      yLabel: figureSettings.yLabel,
      logX: figureSettings.logX,
      logY: figureSettings.logY,
      xMin: figureSettings.xMin,
      xMax: figureSettings.xMax,
      yMin: figureSettings.yMin,
      yMax: figureSettings.yMax,
      showGrid: figureSettings.showGrid,
      gridAlpha: figureSettings.gridAlpha,
      showMinorGrid: figureSettings.showMinorGrid,
      minorGridAlpha: figureSettings.minorGridAlpha,
      showLegend: figureSettings.showLegend,
      legendPosition: figureSettings.legendPosition,
      figWidthInches: figureSettings.figWidthInches,
      figHeightInches: figureSettings.figHeightInches,
      titleFontSize: figureSettings.titleFontSize,
      labelFontSize: figureSettings.labelFontSize,
      legendFontSize: figureSettings.legendFontSize,
      tickFontSizeX: figureSettings.tickFontSizeX,
      tickFontSizeY: figureSettings.tickFontSizeY,
      maxTicksX: figureSettings.maxTicksX,
      maxTicksY: figureSettings.maxTicksY,
      rotateTicksX: figureSettings.rotateTicksX,
      rotateTicksY: figureSettings.rotateTicksY,
    },
    style: matplotlibExportSettings.style,
    export_format: matplotlibExportSettings.format,
    dpi: matplotlibExportSettings.dpi,
  });

  const handleGeneratePreview = async () => {
    if (seriesConfigs.length === 0) {
      setNotification({ type: 'error', message: 'No series to export' });
      return;
    }
    setGeneratingPreview(true);
    try {
      const exportData = {
        ...prepareExportData(),
        export_format: 'png',
        dpi: 150,
      };
      console.log('[ENDFPlotViewer] Generating preview with data:', {
        seriesCount: exportData.series.length,
        style: exportData.style,
        format: exportData.export_format,
        dpi: exportData.dpi,
        seriesSummary: exportData.series.map(s => ({
          fileName: s.file_name,
          mtNumber: s.mt_number,
          order: s.order,
          dataType: s.data_type,
          includeUncertainty: s.include_uncertainty,
          sigma: s.sigma,
        })),
      });
      const result = await exportENDFWithMatplotlib(exportData);
      console.log('[ENDFPlotViewer] Preview generated successfully:', {
        width: result.width_px,
        height: result.height_px,
        dpi: result.dpi,
        format: result.format,
      });
      setPreviewImage({
        base64: result.image_base64,
        format: result.format,
        width: result.width_px,
        height: result.height_px,
        dpi: result.dpi,
      });
    } catch (error) {
      console.error('[ENDFPlotViewer] Preview generation failed:', error);
      setNotification({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to generate preview',
      });
    } finally {
      setGeneratingPreview(false);
    }
  };

  const handleDownloadExport = async () => {
    if (seriesConfigs.length === 0) {
      setNotification({ type: 'error', message: 'No series to export' });
      return;
    }
    setExporting(true);
    try {
      const exportData = prepareExportData();
      console.log('[ENDFPlotViewer] Downloading export with data:', {
        seriesCount: exportData.series.length,
        style: exportData.style,
        format: exportData.export_format,
        dpi: exportData.dpi,
        seriesSummary: exportData.series.map(s => ({
          fileName: s.file_name,
          mtNumber: s.mt_number,
          order: s.order,
          dataType: s.data_type,
          includeUncertainty: s.include_uncertainty,
          sigma: s.sigma,
        })),
      });
      const result = await exportENDFWithMatplotlib(exportData);
      console.log('[ENDFPlotViewer] Export successful:', {
        width: result.width_px,
        height: result.height_px,
        dpi: result.dpi,
        format: result.format,
      });
      const link = document.createElement('a');
      link.href = `data:image/${result.format};base64,${result.image_base64}`;
      link.download = `endf_plot_hq.${result.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setNotification({
        type: 'success',
        message: `High-quality plot downloaded (${result.width_px}×${result.height_px}px @ ${result.dpi} DPI)`,
      });
    } catch (error) {
      console.error('[ENDFPlotViewer] Export failed:', error);
      setNotification({
        type: 'error',
        message: error instanceof Error ? error.message : 'Failed to export plot',
      });
    } finally {
      setExporting(false);
    }
  };

  // Auto-generate preview when dialog opens
  useEffect(() => {
    if (exportDialogOpen && !previewImage && !generatingPreview) {
      handleGeneratePreview();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exportDialogOpen]);

  // Regenerate preview when plot style changes
  useEffect(() => {
    if (exportDialogOpen && !generatingPreview) {
      setPreviewImage(null);
      handleGeneratePreview();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matplotlibExportSettings.style]);

  const disableAddSeries = files.length === 0;

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} lg={4}>
        <Stack spacing={3}>
          <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Alert severity="info" sx={{ mb: 1 }}>
              <Typography variant="body2">
                <strong>⚙️ Server Required:</strong> Ensure the MCNPy processing server is running on <code>localhost:8001</code>
                <br />
                <Typography variant="caption" component="span">
                  Run: <code>python mcnpy_server/app.py</code> in the MCNPy directory
                </Typography>
              </Typography>
            </Alert>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography variant="h6">🧮 Data Mode</Typography>
              <Tooltip title="Reload data">
                <IconButton onClick={() => setSeriesDataMap({})}>
                  <RefreshIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <ToggleButtonGroup
              value={dataMode}
              exclusive
              fullWidth
              onChange={(_event, value) => {
                if (value) {
                  setDataMode(value);
                }
              }}
            >
              {DATA_MODE_OPTIONS.map((option) => (
                <ToggleButton key={option.value} value={option.value} sx={{ textTransform: 'none' }}>
                  {option.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
            <Stack direction="row" spacing={1}>
              <Button
                variant="contained"
                startIcon={<AddCircleOutlineIcon />}
                onClick={handleAddSeries}
                disabled={disableAddSeries}
                fullWidth
              >
                Add Series
              </Button>
              <Button
                variant="outlined"
                color="error"
                onClick={handleClearSeries}
                fullWidth
              >
                Clear All
              </Button>
            </Stack>
            {files.length === 0 ? (
              <Alert severity="info">
                Upload ENDF files in the File Workspace (📁) to get started.
              </Alert>
            ) : files.filter((f) => getAvailableMts(f, dataMode).length > 0).length === 0 ? (
              <Alert severity="warning">
                <strong>No compatible files loaded.</strong>
                <br />
                {dataMode === 'angular'
                  ? 'None of the loaded ENDF files contain MF4 (angular distribution) data. Please upload files with MF4 sections or switch to Uncertainty mode if your files have MF34 data.'
                  : 'None of the loaded ENDF files contain MF34 (uncertainty/covariance) data. Please upload files with MF34 sections or switch to Angular Distributions mode if your files have MF4 data.'}
              </Alert>
            ) : null}
          </Paper>

          {seriesConfigs.map((series, index) => {
            const state = seriesDataMap[series.id];
            const file = files.find((f) => f.name === series.fileName);
            const availableMts = getAvailableMts(file, dataMode);
            const maxOrder = getMaxOrder(file, series.mtNumber);
            const canUseUncertainty =
              dataMode === 'angular' &&
              Boolean(
                file?.metadata?.has_mf34 &&
                  series.mtNumber !== undefined &&
                  file.metadata.uncertainty_mts.includes(series.mtNumber)
              );

            return (
              <Accordion key={series.id} defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                    <Chip size="small" label={`Series ${index + 1}`} color="primary" variant="outlined" />
                    <Typography variant="subtitle1">
                      {series.customLabel || series.autoLabel || state?.data?.label || 'Pending'}
                    </Typography>
                    {state?.status === 'loading' && <LinearProgress sx={{ width: 120 }} />}
                  </Box>
                </AccordionSummary>
                <AccordionDetails sx={{ p: 3 }}>
                  <Stack spacing={2}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="subtitle2">Data Source</Typography>
                      <Tooltip title="Remove series">
                        <IconButton onClick={() => handleRemoveSeries(series.id)}>
                          <DeleteOutlineIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                    <FormControl fullWidth>
                      <InputLabel>ENDF File</InputLabel>
                      <Select
                        value={series.fileName}
                        label="ENDF File"
                        onChange={(event) => {
                          const nextName = event.target.value;
                          const nextFile = files.find((f) => f.name === nextName);
                          const mts = getAvailableMts(nextFile, dataMode);
                          const nextMt = mts[0];
                          const nextOrders = getAvailableOrders(nextFile, dataMode, nextMt);
                          const pickedOrder = nextOrders && nextOrders.length
                            ? (nextOrders.includes(series.order) ? series.order : (nextOrders.includes(1) ? 1 : nextOrders[0]))
                            : Math.min(series.order, getMaxOrder(nextFile, nextMt));
                          handleUpdateSeries(series.id, {
                            fileName: nextName,
                            fileId: nextFile?.metadata?.file_id,
                            mtNumber: nextMt,
                            order: pickedOrder,
                            includeUncertainty:
                              dataMode === 'angular' &&
                              Boolean(
                                nextFile?.metadata?.has_mf34 &&
                                  nextMt !== undefined &&
                                  nextFile.metadata.uncertainty_mts.includes(nextMt)
                              ),
                          });
                        }}
                      >
                        {files.map((fileOption) => (
                          <MenuItem key={fileOption.id} value={fileOption.name}>
                            {fileOption.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <InputLabel>MT Number</InputLabel>
                          <Select
                            value={series.mtNumber ?? ''}
                            label="MT Number"
                            disabled={availableMts.length === 0}
                            onChange={(event) => {
                              const newMt = Number(event.target.value);
                              const nextOrders = getAvailableOrders(file, dataMode, newMt);
                              const pickedOrder = nextOrders.length
                                ? (nextOrders.includes(series.order) ? series.order : (nextOrders.includes(1) ? 1 : nextOrders[0]))
                                : Math.min(series.order, getMaxOrder(file, newMt));
                              handleUpdateSeries(series.id, {
                                mtNumber: newMt,
                                order: pickedOrder,
                                includeUncertainty:
                                  dataMode === 'angular' &&
                                  Boolean(
                                    file?.metadata?.has_mf34 &&
                                      file.metadata.uncertainty_mts.includes(newMt)
                                  ),
                              });
                            }}
                          >
                            {availableMts.map((mt) => (
                              <MenuItem key={mt} value={mt}>
                                MT {mt}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Grid>
                      <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                          <InputLabel>Legendre Order (L)</InputLabel>
                          <Select
                            value={series.order}
                            label="Legendre Order (L)"
                            disabled={(getAvailableOrders(file, dataMode, series.mtNumber).length === 0) && maxOrder === 0}
                            onChange={(event) => {
                              handleUpdateSeries(series.id, { order: Number(event.target.value) });
                            }}
                          >
                            {getAvailableOrders(file, dataMode, series.mtNumber).map((order) => (
                              <MenuItem key={order} value={order}>
                                L = {order} {order === 0 ? '(isotropic)' : ''}
                              </MenuItem>
                            ))}
                            {getAvailableOrders(file, dataMode, series.mtNumber).length === 0 &&
                              Array.from({ length: maxOrder + 1 }, (_, i) => i).map((order) => (
                                <MenuItem key={order} value={order}>
                                  L = {order} {order === 0 ? '(isotropic)' : ''}
                                </MenuItem>
                              ))}
                          </Select>
                        </FormControl>
                      </Grid>
                    </Grid>

                    {availableMts.length === 0 && (
                      <Alert severity="warning">
                        {dataMode === 'angular' 
                          ? 'This file does not contain MF4 (angular distribution) data. Switch to a file with MF4 data or upload a different ENDF file.'
                          : 'This file does not contain MF34 (uncertainty) data. Switch to a file with MF34 covariance data or upload a different ENDF file.'}
                      </Alert>
                    )}

                    {state?.status === 'error' && (
                      <Alert severity="error">
                        <strong>Error loading data:</strong> {state.error}
                        <br />
                        <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                          Possible causes:
                          <br />
                          • <strong>Server not running:</strong> Ensure the MCNPy server is running on localhost:8001 (run <code>python mcnpy_server/app.py</code>)
                          <br />
                          • <strong>Invalid MT/Order:</strong> The selected MT number or Legendre order may not exist in this file
                          <br />
                          • <strong>Missing data:</strong> The file may not contain the requested data type (MF4 or MF34)
                        </Typography>
                      </Alert>
                    )}

                    {dataMode === 'angular' && (
                      <Box>
                        <Tooltip 
                          title={
                            canUseUncertainty 
                              ? 'Include MF34 uncertainty/covariance data as a shaded band around the curve' 
                              : `This ENDF file does not contain MF34 (uncertainty/covariance) data for MT${series.mtNumber}. Only files with MF34 sections can display uncertainty bands.`
                          }
                          arrow
                        >
                          <FormControlLabel
                            control={
                              <Switch
                                checked={series.includeUncertainty && canUseUncertainty}
                                onChange={(_event, checked) =>
                                  handleUpdateSeries(series.id, { includeUncertainty: checked && canUseUncertainty })
                                }
                                disabled={!canUseUncertainty}
                              />
                            }
                            label={
                              canUseUncertainty
                                ? 'Include uncertainty band (MF34)'
                                : '⚠️ No MF34 uncertainty data for this MT'
                            }
                          />
                        </Tooltip>
                        {!canUseUncertainty && (
                          <Alert severity="info" sx={{ mt: 1 }}>
                            <Typography variant="body2">
                              <strong>Note:</strong> This ENDF file doesn't have MF34 covariance data for MT{series.mtNumber}.
                              <br />
                              <Typography variant="caption" component="span">
                                • MF4 contains angular distributions (what you're plotting)
                                <br />
                                • MF34 contains uncertainties for MF4 (currently missing)
                                <br />
                                • Libraries like TENDL often include MF34 data
                              </Typography>
                            </Typography>
                          </Alert>
                        )}
                        {canUseUncertainty && series.includeUncertainty && state?.status === 'ready' && !state.data?.uncertainty && (
                          <Alert severity="warning" sx={{ mt: 1 }}>
                            <Typography variant="body2">
                              <strong>Backend Issue:</strong> Uncertainty was requested but not returned by the server.
                              <br />
                              <Typography variant="caption" component="span">
                                This could indicate:
                                <br />
                                • The MCNPy library couldn't find MF34 data for this specific order/MT combination
                                <br />
                                • There's a parsing error in the ENDF file
                                <br />
                                • Check the server console for error messages
                              </Typography>
                            </Typography>
                          </Alert>
                        )}
                      </Box>
                    )}

                    {dataMode === 'angular' && series.includeUncertainty && canUseUncertainty && (
                      <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                          <Typography variant="caption" sx={{ mr: 0.5 }}>Sigma (σ)</Typography>
                          <Tooltip title={'Approximate normal coverage: 1σ ≈ 68.3%, 2σ ≈ 95.4%, 3σ ≈ 99.7%'}>
                            <InfoOutlinedIcon fontSize="inherit" color="action" />
                          </Tooltip>
                        </Box>
                        <TextField
                          type="number"
                          value={series.sigma}
                          onChange={(event) => {
                            const value = Number(event.target.value);
                            if (Number.isNaN(value) || value <= 0) return;
                            handleUpdateSeries(series.id, { sigma: value });
                          }}
                          inputProps={{ min: 0.5, max: 3, step: 0.5 }}
                          fullWidth
                        />
                      </Box>
                    )}

                    <Divider />

                    <Typography variant="subtitle2">Label</Typography>
                    <ToggleButtonGroup
                      value={series.labelMode}
                      exclusive
                      onChange={(_event, value) => value && handleUpdateSeries(series.id, { labelMode: value })}
                      size="small"
                    >
                      <ToggleButton value="auto">Auto</ToggleButton>
                      <ToggleButton value="custom">Custom</ToggleButton>
                    </ToggleButtonGroup>
                    {series.labelMode === 'custom' && (
                      <TextField
                        label="Custom Label"
                        value={series.customLabel}
                        onChange={(event) => handleUpdateSeries(series.id, { customLabel: event.target.value })}
                        fullWidth
                      />
                    )}

                    <Divider sx={{ mt: 1 }} />
                    
                    {/* Collapsible Styling Section - matching PlotViewer exactly */}
                    <Accordion disableGutters elevation={0} sx={{ '&:before': { display: 'none' } }}>
                      <AccordionSummary 
                        expandIcon={<ExpandMoreIcon />}
                        sx={{ px: 0, minHeight: 40, '&.Mui-expanded': { minHeight: 40 } }}
                      >
                        <Typography variant="subtitle2">🎨 Styling Options</Typography>
                      </AccordionSummary>
                      <AccordionDetails sx={{ p: 0 }}>
                        <Stack spacing={2}>
                          <Typography variant="caption" color="text.secondary">Color</Typography>
                          <Stack direction="row" flexWrap="wrap" spacing={1}>
                            {COLOR_PALETTE.map((color) => (
                              <Box
                                key={color}
                                onClick={() => handleUpdateSeries(series.id, { color })}
                                sx={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: '50%',
                                  border: series.color === color ? '2px solid #000' : '1px solid #ccc',
                                  backgroundColor: color,
                                  cursor: 'pointer',
                                  '&:hover': {
                                    transform: 'scale(1.1)',
                                    transition: 'transform 0.2s',
                                  },
                                }}
                              />
                            ))}
                          </Stack>
                          <TextField
                            label="Custom Color"
                            type="color"
                            value={series.color}
                            onChange={(event) =>
                              handleUpdateSeries(series.id, { color: event.target.value })
                            }
                            sx={{ width: 140 }}
                            size="small"
                          />
                          <Grid container spacing={2}>
                            <Grid item xs={6}>
                              <TextField
                                label="Line Width"
                                type="number"
                                value={series.lineWidth}
                                onChange={(event) =>
                                  handleUpdateSeries(series.id, { lineWidth: Number(event.target.value) })
                                }
                                inputProps={{ min: 0.5, step: 0.5, max: 8 }}
                                fullWidth
                                size="small"
                              />
                            </Grid>
                            <Grid item xs={6}>
                              <FormControl fullWidth size="small">
                                <InputLabel>Line Style</InputLabel>
                                <Select
                                  value={series.lineStyle}
                                  label="Line Style"
                                  onChange={(event) =>
                                    handleUpdateSeries(series.id, {
                                      lineStyle: event.target.value as SeriesConfig['lineStyle'],
                                    })
                                  }
                                >
                                  {LINE_STYLE_OPTIONS.map((style) => (
                                    <MenuItem key={style.value} value={style.value}>
                                      {style.label}
                                    </MenuItem>
                                  ))}
                                </Select>
                              </FormControl>
                            </Grid>
                          </Grid>

                          <FormControlLabel
                            control={
                              <Switch
                                checked={series.showMarkers}
                                onChange={(event) =>
                                  handleUpdateSeries(series.id, { showMarkers: event.target.checked })
                                }
                              />
                            }
                            label="Show markers"
                          />

                          {series.showMarkers && (
                            <Grid container spacing={2}>
                              <Grid item xs={6}>
                                <FormControl fullWidth size="small">
                                  <InputLabel>Marker</InputLabel>
                                  <Select
                                    value={series.markerSymbol}
                                    label="Marker"
                                    onChange={(event) =>
                                      handleUpdateSeries(series.id, {
                                        markerSymbol: event.target.value as PlotlyMarkerSymbol,
                                      })
                                    }
                                  >
                                    {MARKER_OPTIONS.map((marker) => (
                                      <MenuItem key={marker.value as string} value={marker.value as string}>
                                        {marker.label}
                                      </MenuItem>
                                    ))}
                                  </Select>
                                </FormControl>
                              </Grid>
                              <Grid item xs={6}>
                                <TextField
                                  label="Marker Size"
                                  type="number"
                                  value={series.markerSize}
                                  onChange={(event) =>
                                    handleUpdateSeries(series.id, { markerSize: Number(event.target.value) })
                                  }
                                  inputProps={{ min: 2, max: 20 }}
                                  fullWidth
                                  size="small"
                                />
                              </Grid>
                            </Grid>
                          )}
                        </Stack>
                      </AccordionDetails>
                    </Accordion>
                  </Stack>
                </AccordionDetails>
              </Accordion>
            );
          })}

          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">🎛️ Figure Settings</Typography>
              <Button
                size="small"
                startIcon={<RefreshIcon fontSize="small" />}
                onClick={() => setFigureSettings(getDefaultFigureSettings())}
              >
                Reset All
              </Button>
            </Box>

            {/* Labels, Title & Legend */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">🏷️ Labels, Title & Legend</Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 4 }}>
                <Stack spacing={2}>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={9}>
                      <TextField
                        label="Title"
                        value={figureSettings.title}
                        onChange={(event) => setFigureSettings({ ...figureSettings, title: event.target.value })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={3}>
                      <TextField
                        label="Size"
                        type="number"
                        value={figureSettings.titleFontSize}
                        onChange={(event) => setFigureSettings({ ...figureSettings, titleFontSize: Number(event.target.value) })}
                        inputProps={{ min: 8, max: 32, step: 1 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>

                  <Typography variant="subtitle2" sx={{ mt: 1 }}>Axis Labels</Typography>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={9}>
                      <TextField
                        label="X-axis"
                        value={figureSettings.xLabel}
                        onChange={(event) => setFigureSettings({ ...figureSettings, xLabel: event.target.value })}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={3}>
                      <TextField
                        label="Size"
                        type="number"
                        value={figureSettings.labelFontSize}
                        onChange={(event) => setFigureSettings({ ...figureSettings, labelFontSize: Number(event.target.value) })}
                        inputProps={{ min: 8, max: 24, step: 1 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        label="Y-axis"
                        value={figureSettings.yLabel}
                        onChange={(event) => setFigureSettings({ ...figureSettings, yLabel: event.target.value })}
                        fullWidth
                      />
                    </Grid>
                  </Grid>

                  <Typography variant="subtitle2" sx={{ mt: 1 }}>Legend</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={figureSettings.showLegend}
                            onChange={(event) => setFigureSettings({ ...figureSettings, showLegend: event.target.checked })}
                          />
                        }
                        label="Show"
                      />
                    </Grid>
                    <Grid item xs={5}>
                      <FormControl fullWidth disabled={!figureSettings.showLegend}>
                        <InputLabel>Position</InputLabel>
                        <Select
                          value={figureSettings.legendPosition}
                          label="Position"
                          onChange={(event) => setFigureSettings({ ...figureSettings, legendPosition: event.target.value as FigureSettings['legendPosition'] })}
                        >
                          <MenuItem value="top-right">Top Right</MenuItem>
                          <MenuItem value="top-left">Top Left</MenuItem>
                          <MenuItem value="bottom-right">Bottom Right</MenuItem>
                          <MenuItem value="bottom-left">Bottom Left</MenuItem>
                          <MenuItem value="outside">Outside (Right)</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={3}>
                      <TextField
                        label="Size"
                        type="number"
                        value={figureSettings.legendFontSize}
                        onChange={(event) => setFigureSettings({ ...figureSettings, legendFontSize: Number(event.target.value) })}
                        inputProps={{ min: 6, max: 20, step: 1 }}
                        fullWidth
                        disabled={!figureSettings.showLegend}
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </AccordionDetails>
            </Accordion>

            {/* Zoom, Scales & Grid */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">🔍 Zoom, Scales & Grid</Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 4 }}>
                <Stack spacing={2}>
                  <Typography variant="subtitle2">🔍 Zoom / Axis Limits</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">X-axis Range</Typography>
                      <Grid container spacing={1} sx={{ mt: 0.5 }}>
                        <Grid item xs={6}>
                          <TextField
                            label="Min"
                            placeholder="auto"
                            value={figureSettings.xMin}
                            onChange={(event) => setFigureSettings({ ...figureSettings, xMin: event.target.value })}
                            fullWidth
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            label="Max"
                            placeholder="auto"
                            value={figureSettings.xMax}
                            onChange={(event) => setFigureSettings({ ...figureSettings, xMax: event.target.value })}
                            fullWidth
                            size="small"
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="caption" color="text.secondary">Y-axis Range</Typography>
                      <Grid container spacing={1} sx={{ mt: 0.5 }}>
                        <Grid item xs={6}>
                          <TextField
                            label="Min"
                            placeholder="auto"
                            value={figureSettings.yMin}
                            onChange={(event) => setFigureSettings({ ...figureSettings, yMin: event.target.value })}
                            fullWidth
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            label="Max"
                            placeholder="auto"
                            value={figureSettings.yMax}
                            onChange={(event) => setFigureSettings({ ...figureSettings, yMax: event.target.value })}
                            fullWidth
                            size="small"
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                  </Grid>

                  <Divider sx={{ my: 1 }} />

                  <Typography variant="subtitle2">Scales</Typography>
                  <Stack direction="row" spacing={3}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={figureSettings.logX}
                          onChange={(event) => setFigureSettings({ ...figureSettings, logX: event.target.checked })}
                        />
                      }
                      label="Log X"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={figureSettings.logY}
                          onChange={(event) => setFigureSettings({ ...figureSettings, logY: event.target.checked })}
                        />
                      }
                      label="Log Y"
                    />
                  </Stack>

                  <Typography variant="subtitle2">Grid</Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={figureSettings.showGrid}
                        onChange={(event) => setFigureSettings({ ...figureSettings, showGrid: event.target.checked })}
                      />
                    }
                    label="Show grid"
                  />
                  {figureSettings.showGrid && (
                    <Box>
                      <Typography variant="caption">Grid transparency: {figureSettings.gridAlpha.toFixed(2)}</Typography>
                      <Box sx={{ px: 1 }}>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={figureSettings.gridAlpha}
                          onChange={(event) => setFigureSettings({ ...figureSettings, gridAlpha: Number(event.target.value) })}
                          style={{ width: '100%' }}
                        />
                      </Box>
                      
                      {/* Minor Grid */}
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={figureSettings.showMinorGrid}
                              onChange={(event) => setFigureSettings({ ...figureSettings, showMinorGrid: event.target.checked })}
                            />
                          }
                          label="Show minor grid"
                        />
                        <Tooltip title="Minor grid is only visible in High Quality Export (not in Plotly preview)">
                          <InfoOutlinedIcon fontSize="small" color="action" sx={{ ml: 0.5 }} />
                        </Tooltip>
                      </Box>
                      
                      {figureSettings.showMinorGrid && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption">Minor grid transparency: {figureSettings.minorGridAlpha.toFixed(2)}</Typography>
                          <Box sx={{ px: 1 }}>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={figureSettings.minorGridAlpha}
                              onChange={(event) => setFigureSettings({ ...figureSettings, minorGridAlpha: Number(event.target.value) })}
                              style={{ width: '100%' }}
                            />
                          </Box>
                        </Box>
                      )}
                    </Box>
                  )}
                </Stack>
              </AccordionDetails>
            </Accordion>

            {/* Figure Size */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">📐 Figure Size</Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 4 }}>
                <Stack spacing={2}>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <TextField
                        label="Width (inches)"
                        type="number"
                        value={figureSettings.figWidthInches}
                        onChange={(event) => setFigureSettings({ ...figureSettings, figWidthInches: Number(event.target.value) })}
                        inputProps={{ min: 4, max: 20, step: 0.5 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <TextField
                        label="Height (inches)"
                        type="number"
                        value={figureSettings.figHeightInches}
                        onChange={(event) => setFigureSettings({ ...figureSettings, figHeightInches: Number(event.target.value) })}
                        inputProps={{ min: 3, max: 16, step: 0.5 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>
                  <Typography variant="caption" color="text.secondary">
                    Display: {Math.round(figureSettings.figWidthInches * 96)} × {Math.round(figureSettings.figHeightInches * 96)} pixels @ 96 DPI
                  </Typography>
                </Stack>
              </AccordionDetails>
            </Accordion>

            {/* Advanced: Tick Parameters */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1">⚙️ Advanced: Tick Parameters</Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ p: 4 }}>
                <Stack spacing={2}>
                  <Typography variant="subtitle2">X-axis Ticks</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <TextField
                        label="Font Size"
                        type="number"
                        value={figureSettings.tickFontSizeX}
                        onChange={(event) => setFigureSettings({ ...figureSettings, tickFontSizeX: Number(event.target.value) })}
                        inputProps={{ min: 6, max: 20, step: 1 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <TextField
                        label="Max Ticks"
                        type="number"
                        value={figureSettings.maxTicksX}
                        onChange={(event) => setFigureSettings({ ...figureSettings, maxTicksX: Number(event.target.value) })}
                        inputProps={{ min: 3, max: 20, step: 1 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <TextField
                        label="Rotation (°)"
                        type="number"
                        value={figureSettings.rotateTicksX}
                        onChange={(event) => setFigureSettings({ ...figureSettings, rotateTicksX: Number(event.target.value) })}
                        inputProps={{ min: 0, max: 90, step: 15 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>

                  <Typography variant="subtitle2" sx={{ mt: 1 }}>Y-axis Ticks</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={4}>
                      <TextField
                        label="Font Size"
                        type="number"
                        value={figureSettings.tickFontSizeY}
                        onChange={(event) => setFigureSettings({ ...figureSettings, tickFontSizeY: Number(event.target.value) })}
                        inputProps={{ min: 6, max: 20, step: 1 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <TextField
                        label="Max Ticks"
                        type="number"
                        value={figureSettings.maxTicksY}
                        onChange={(event) => setFigureSettings({ ...figureSettings, maxTicksY: Number(event.target.value) })}
                        inputProps={{ min: 3, max: 20, step: 1 }}
                        fullWidth
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <TextField
                        label="Rotation (°)"
                        type="number"
                        value={figureSettings.rotateTicksY}
                        onChange={(event) => setFigureSettings({ ...figureSettings, rotateTicksY: Number(event.target.value) })}
                        inputProps={{ min: 0, max: 90, step: 15 }}
                        fullWidth
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </AccordionDetails>
            </Accordion>
          </Paper>

          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Saved Configurations
            </Typography>
            <Stack spacing={2}>
              <TextField
                label="Configuration Name"
                value={configName}
                onChange={(event) => setConfigName(event.target.value)}
              />
              <Button
                variant="contained"
                startIcon={<SaveOutlinedIcon />}
                onClick={handleSaveConfig}
                disabled={files.length === 0}
              >
                Save Current Setup
              </Button>
              {savedConfigs.length === 0 && (
                <Alert severity="info">No saved plots yet. Save your layout to reuse later.</Alert>
              )}
              {savedConfigs.map((configInfo) => (
                <Paper key={configInfo.id} variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle1">{configInfo.name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {configInfo.dataMode === 'angular' ? 'Angular Distributions' : 'Uncertainties'} • {formatTimestamp(configInfo.createdAt)}
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                    <Button
                      startIcon={<RestoreOutlinedIcon />}
                      size="small"
                      onClick={() => handleRestoreConfig(configInfo.id)}
                    >
                      Restore
                    </Button>
                    <Button
                      size="small"
                      color="error"
                      onClick={() => handleDeleteConfig(configInfo.id)}
                    >
                      Delete
                    </Button>
                  </Stack>
                </Paper>
              ))}
            </Stack>
          </Paper>

          {notification && (
            <Alert severity={notification.type} onClose={() => setNotification(null)} sx={{ mt: 2 }}>
              {notification.message}
            </Alert>
          )}
        </Stack>
      </Grid>

      <Grid item xs={12} lg={8}>
        <Box sx={{ position: 'sticky', top: 16, maxHeight: 'calc(100vh - 32px)', overflowY: 'auto' }}>
          <Paper sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Typography variant="h6">📊 Plot Preview</Typography>
            {plotlyData.length === 0 ? (
              <Box
                sx={{
                  minHeight: 400,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '2px dashed',
                  borderColor: 'grey.300',
                  borderRadius: 2,
                }}
              >
                <Typography color="text.secondary" align="center">
                  Configure series on the left and the plot will appear here in real time.
                </Typography>
              </Box>
            ) : (
              <Box
                sx={{
                  width: '100%',
                  maxWidth: '100%',
                  overflow: 'auto',
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                }}
              >
                <Plot
                  data={plotlyData}
                  layout={layout}
                  config={config}
                  style={{ width: '100%', height: '100%' }}
                  onInitialized={(_figure, graphDiv) => {
                    plotRef.current = graphDiv as PlotlyHTMLElement;
                  }}
                  onUpdate={(_figure, graphDiv) => {
                    plotRef.current = graphDiv as PlotlyHTMLElement;
                  }}
                />
              </Box>
            )}

            <Stack direction="row" spacing={2}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleQuickExport}
                disabled={exporting || plotlyData.length === 0}
                fullWidth
              >
                {exporting ? 'Exporting...' : 'Quick Export (Plotly)'}
              </Button>
              <Button
                variant="contained"
                startIcon={<HighQualityIcon />}
                onClick={() => setExportDialogOpen(true)}
                disabled={exporting || plotlyData.length === 0}
                fullWidth
                color="secondary"
              >
                Export (High Quality)
              </Button>
            </Stack>
          </Paper>
        </Box>
      </Grid>

      <Dialog
        fullWidth
        maxWidth="md"
        open={exportDialogOpen}
        onClose={() => {
          setExportDialogOpen(false);
          setPreviewImage(null);
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <HighQualityIcon /> High-quality Matplotlib Export
        </DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={3}>
            <Grid item xs={12} md={5}>
              <Stack spacing={2}>
                <FormControl fullWidth>
                  <InputLabel>Plot Style</InputLabel>
                  <Select
                    value={matplotlibExportSettings.style}
                    label="Plot Style"
                    onChange={(event) =>
                      setMatplotlibExportSettings({ ...matplotlibExportSettings, style: event.target.value })
                    }
                  >
                    <MenuItem value="publication">Publication</MenuItem>
                    <MenuItem value="default">Default</MenuItem>
                    <MenuItem value="presentation">Presentation</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                  </Select>
                </FormControl>
                <FormControl fullWidth>
                  <InputLabel>Export Format</InputLabel>
                  <Select
                    value={matplotlibExportSettings.format}
                    label="Export Format"
                    onChange={(event) =>
                      setMatplotlibExportSettings({ ...matplotlibExportSettings, format: event.target.value })
                    }
                  >
                    <MenuItem value="png">PNG</MenuItem>
                    <MenuItem value="pdf">PDF</MenuItem>
                    <MenuItem value="svg">SVG</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  label="DPI"
                  type="number"
                  value={matplotlibExportSettings.dpi}
                  onChange={(event) =>
                    setMatplotlibExportSettings({ ...matplotlibExportSettings, dpi: Number(event.target.value) })
                  }
                  inputProps={{ min: 150, max: 600, step: 25 }}
                />
                <Button
                  variant="outlined"
                  startIcon={<InfoOutlinedIcon />}
                  onClick={handleGeneratePreview}
                  disabled={generatingPreview}
                >
                  Refresh Preview
                </Button>
              </Stack>
            </Grid>
            <Grid item xs={12} md={7}>
              <Box
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  minHeight: 320,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {generatingPreview && (
                  <Box sx={{ textAlign: 'center' }}>
                    <LinearProgress sx={{ width: '60%', mb: 2 }} />
                    <Typography variant="body2" color="text.secondary">
                      Generating preview...
                    </Typography>
                  </Box>
                )}
                {!generatingPreview && previewImage && (
                  <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <img
                      src={`data:image/${previewImage.format};base64,${previewImage.base64}`}
                      alt="Matplotlib preview"
                      style={{ maxWidth: '100%', height: 'auto' }}
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                      Preview: {previewImage.width} × {previewImage.height} px @ {previewImage.dpi} DPI
                    </Typography>
                  </Box>
                )}
                {!generatingPreview && !previewImage && (
                  <Alert severity="info" sx={{ m: 2 }}>
                    Click "Refresh Preview" or change the plot style to generate a preview.
                  </Alert>
                )}
              </Box>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportDialogOpen(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={handleDownloadExport}
            startIcon={<DownloadIcon />}
            disabled={exporting}
          >
            Download
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};
