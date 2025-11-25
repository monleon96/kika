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
import { fetchSeriesData, exportWithMatplotlib, type ACEInfo, type SeriesDataResponse } from '../services/mcnpyService';

type PlotType = 'xs' | 'angular';

export interface LoadedACEFile {
  name: string;
  path: string;
  content: string;
  info?: ACEInfo;
  error?: string;
}

interface PlotViewerProps {
  files: LoadedACEFile[];
}

interface SeriesConfig {
  id: string;
  fileName: string;
  fileId?: string;
  mtNumber?: number;
  energy?: number;
  labelMode: 'auto' | 'custom';
  customLabel: string;
  color: string;
  lineWidth: number;
  lineStyle: 'solid' | 'dash' | 'dot' | 'dashdot';
  showMarkers: boolean;
  markerSymbol: PlotlyMarkerSymbol;
  markerSize: number;
  autoLabel?: string;
}

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

interface SeriesDataState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  signature?: string;
  data?: SeriesDataResponse;
  error?: string;
}

interface SavedPlotConfig {
  id: string;
  name: string;
  createdAt: string;
  plotType: PlotType;
  figure: FigureSettings;
  series: SeriesConfig[];
}

// Color-blind friendly palette matching Matplotlib publication style
const COLOR_PALETTE = [
  '#0173B2', // Blue
  '#DE8F05', // Orange
  '#029E73', // Green
  '#D55E00', // Red-orange
  '#CC78BC', // Purple
  '#CA9161', // Brown
  '#FBAFE4', // Pink
  '#949494', // Gray
  '#ECE133', // Yellow
  '#56B4E9', // Light blue
];

const LINE_STYLE_OPTIONS: { label: string; value: SeriesConfig['lineStyle'] }[] = [
  { label: 'Solid', value: 'solid' },
  { label: 'Dashed', value: 'dash' },
  { label: 'Dotted', value: 'dot' },
  { label: 'Dash-dot', value: 'dashdot' },
];

const MARKER_OPTIONS: { label: string; value: PlotlyMarkerSymbol }[] = [
  { label: 'Circle', value: 'circle' },
  { label: 'Square', value: 'square' },
  { label: 'Triangle up', value: 'triangle-up' },
  { label: 'Triangle down', value: 'triangle-down' },
  { label: 'Diamond', value: 'diamond' },
  { label: 'Star', value: 'star' },
  { label: 'X', value: 'x' },
  { label: 'Plus', value: 'cross' },
];

const DEFAULT_ANGULAR_ENERGY = 1.0;
const LOCAL_STORAGE_KEY = 'kikaAceViewerConfigs';

const createSeriesId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const getDefaultFigureSettings = (plotType: PlotType): FigureSettings => ({
  title: plotType === 'xs' ? 'Cross Section Comparison' : 'Angular Distribution Comparison',
  xLabel: plotType === 'xs' ? 'Energy (MeV)' : 'cos(theta)',
  yLabel: plotType === 'xs' ? 'Cross Section (barns)' : 'Probability Density',
  logX: plotType === 'xs',
  logY: plotType === 'xs',
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

const getAvailableMts = (file: LoadedACEFile | undefined, plotType: PlotType): number[] => {
  if (!file?.info) return [];
  if (plotType === 'xs') {
    return file.info.available_reactions || [];
  }
  if (file.info.angular_reactions && file.info.angular_reactions.length > 0) {
    return file.info.angular_reactions;
  }
  return [];
};

const formatTimestamp = (iso: string) => {
  const date = new Date(iso);
  return date.toLocaleString();
};

export const PlotViewer: React.FC<PlotViewerProps> = ({ files }) => {
  const [plotType, setPlotType] = useState<PlotType>('xs');
  const [figureSettings, setFigureSettings] = useState<FigureSettings>(() => getDefaultFigureSettings('xs'));
  const [seriesConfigs, setSeriesConfigs] = useState<SeriesConfig[]>([]);
  const [seriesDataMap, setSeriesDataMap] = useState<Record<string, SeriesDataState>>({});
  const [notification, setNotification] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [configName, setConfigName] = useState('');
  // Do NOT persist saved configs - they should be session-only
  const [savedConfigs, setSavedConfigs] = useState<SavedPlotConfig[]>([]);
  const [exportOptions] = useState({ format: 'png', width: 1200, height: 720, scale: 2 });
  const [exporting, setExporting] = useState(false);
  const plotRef = useRef<PlotlyHTMLElement | null>(null);
  const [showMatplotlibExport, setShowMatplotlibExport] = useState(false);
  const [matplotlibExportSettings, setMatplotlibExportSettings] = useState({
    style: 'publication',
    format: 'png',
    dpi: 300,
  });
  const [previewImage, setPreviewImage] = useState<{
    base64: string;
    format: string;
    width: number;
    height: number;
    dpi: number;
  } | null>(null);
  const [generatingPreview, setGeneratingPreview] = useState(false);

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
    setSeriesConfigs((prev) =>
      prev.map((series) => {
        const file = files.find((f) => f.name === series.fileName);
        if (file?.info?.file_id && series.fileId !== file.info.file_id) {
          return { ...series, fileId: file.info.file_id };
        }
        return series;
      }),
    );
  }, [files]);

  const updatePlotType = (nextType: PlotType, options?: { preserveState?: boolean }) => {
    setPlotType(nextType);
    if (options?.preserveState) {
      return;
    }
    setSeriesConfigs([]);
    setSeriesDataMap({});
    setFigureSettings(getDefaultFigureSettings(nextType));
  };

  const handleAddSeries = () => {
    console.log('[PlotViewer] handleAddSeries called, files:', files.length);
    if (files.length === 0) {
      console.log('[PlotViewer] No files available');
      return;
    }
    const defaultFile = files[0];
    console.log('[PlotViewer] Default file:', defaultFile.name, 'has info:', !!defaultFile.info);
    const mtList = getAvailableMts(defaultFile, plotType);
    console.log('[PlotViewer] Available MTs:', mtList);
    const nextSeries: SeriesConfig = {
      id: createSeriesId(),
      fileName: defaultFile.name,
      fileId: defaultFile.info?.file_id,
      mtNumber: mtList.length > 0 ? (mtList.includes(2) ? 2 : mtList[0]) : undefined,
      energy: plotType === 'angular' ? DEFAULT_ANGULAR_ENERGY : undefined,
      labelMode: 'auto',
      customLabel: '',
      color: COLOR_PALETTE[seriesConfigs.length % COLOR_PALETTE.length],
      lineWidth: 2,
      lineStyle: 'solid',
      showMarkers: false,
      markerSymbol: 'circle',
      markerSize: 8,
    };
    console.log('[PlotViewer] Creating series:', nextSeries);
    setSeriesConfigs((prev) => [...prev, nextSeries]);
  };

  const handleUpdateSeries = (id: string, patch: Partial<SeriesConfig>) => {
    setSeriesConfigs((prev) =>
      prev.map((series) => (series.id === id ? { ...series, ...patch } : series)),
    );
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

  useEffect(() => {
    seriesConfigs.forEach((series) => {
      if (!series.fileName || series.mtNumber === undefined) return;
      const file = files.find((f) => f.name === series.fileName);
      if (!file) return;
      const fileId = file.info?.file_id;
      const signatureParts = [
        fileId ?? file.content.slice(0, 50),
        plotType,
        series.mtNumber,
        plotType === 'angular' ? series.energy ?? DEFAULT_ANGULAR_ENERGY : 'xs',
      ];
      const signature = signatureParts.join(':');
      const existing = seriesDataMap[series.id];
      if (existing?.signature === signature && (existing.status === 'ready' || existing.status === 'loading')) {
        return;
      }

      setSeriesDataMap((prev) => ({
        ...prev,
        [series.id]: { status: 'loading', signature },
      }));

      fetchSeriesData({
        file_id: fileId,
        file_content: fileId ? undefined : file.content,
        file_name: file.name,
        plot_type: plotType,
        mt_number: series.mtNumber,
        energy: plotType === 'angular' ? series.energy ?? DEFAULT_ANGULAR_ENERGY : undefined,
      })
        .then((data) => {
          console.log('[PlotViewer] Received data from server:', {
            seriesId: series.id,
            label: data.label,
            xLength: data.x?.length,
            yLength: data.y?.length,
            xType: typeof data.x,
            yType: typeof data.y,
            xIsArray: Array.isArray(data.x),
            yIsArray: Array.isArray(data.y),
            xSample: data.x?.slice?.(0, 3),
            ySample: data.y?.slice?.(0, 3),
          });
          setSeriesDataMap((prev) => ({
            ...prev,
            [series.id]: { status: 'ready', signature, data },
          }));
          setSeriesConfigs((prev) =>
            prev.map((cfg) =>
              cfg.id === series.id
                ? {
                    ...cfg,
                    fileId: file.info?.file_id,
                    autoLabel: data.label,
                  }
                : cfg,
            ),
          );
        })
        .catch((error) => {
          console.error('[PlotViewer] Error fetching series data:', error);
          setSeriesDataMap((prev) => ({
            ...prev,
            [series.id]: { status: 'error', signature, error: error instanceof Error ? error.message : 'Failed to load data' },
          }));
        });
    });
  }, [seriesConfigs, files, plotType, seriesDataMap]);

  const plotlyData: PlotTrace[] = useMemo(() => {
    const traces = seriesConfigs
      .map((series, index) => {
        const state = seriesDataMap[series.id];
        if (!state || state.status !== 'ready' || !state.data) return null;
        const label =
          series.labelMode === 'custom' && series.customLabel.trim().length > 0
            ? series.customLabel.trim()
            : series.autoLabel || state.data.label || `Series ${index + 1}`;

        const traceInfo: any = {
          seriesId: series.id,
          label,
          xLength: state.data.x?.length || 0,
          yLength: state.data.y?.length || 0,
          xType: typeof state.data.x,
          yType: typeof state.data.y,
          xIsArray: Array.isArray(state.data.x),
          yIsArray: Array.isArray(state.data.y),
        };
        
        if (state.data.x?.length > 0 && state.data.y?.length > 0) {
          const xArr = Array.isArray(state.data.x) ? state.data.x : [];
          const yArr = Array.isArray(state.data.y) ? state.data.y : [];
          traceInfo.xSample = xArr.slice(0, 3);
          traceInfo.ySample = yArr.slice(0, 3);
          traceInfo.xMin = Math.min(...xArr);
          traceInfo.xMax = Math.max(...xArr);
          traceInfo.yMin = Math.min(...yArr);
          traceInfo.yMax = Math.max(...yArr);
        }
        
        console.log('[PlotViewer] Creating trace:', traceInfo.label);
        console.log('  Data points:', traceInfo.xLength, 'x', traceInfo.yLength);
        if (traceInfo.xMin !== undefined) {
          console.log('  X range:', traceInfo.xMin, 'to', traceInfo.xMax);
          console.log('  Y range:', traceInfo.yMin, 'to', traceInfo.yMax);
          console.log('  X sample:', traceInfo.xSample);
          console.log('  Y sample:', traceInfo.ySample);
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
          },
        };
        
        if (series.showMarkers) {
          trace.marker = {
            color: series.color,
            size: series.markerSize,
            symbol: series.markerSymbol,
          };
        }
        
        return trace;
      })
      .filter(Boolean) as PlotTrace[];
    
    console.log('[PlotViewer] Total traces:', traces.length);
    if (traces.length > 0) {
      console.log('[PlotViewer] First trace passed to Plotly:', {
        name: traces[0].name,
        type: traces[0].type,
        mode: traces[0].mode,
        xLength: traces[0].x?.length,
        yLength: traces[0].y?.length,
        xFirst3: Array.isArray(traces[0].x) ? traces[0].x.slice(0, 3) : 'not array',
        yFirst3: Array.isArray(traces[0].y) ? traces[0].y.slice(0, 3) : 'not array',
      });
    }
    return traces;
  }, [seriesConfigs, seriesDataMap]);

  const parseRange = (minStr: string, maxStr: string, isLog: boolean): [number, number] | undefined => {
    if (!minStr && !maxStr) return undefined;
    const min = minStr ? Number(minStr) : undefined;
    const max = maxStr ? Number(maxStr) : undefined;
    if ((minStr && Number.isNaN(min)) || (maxStr && Number.isNaN(max))) {
      return undefined;
    }
    if (min !== undefined && max !== undefined) {
      if (isLog && (min <= 0 || max <= 0)) return undefined;
      // For log scale, Plotly expects log10 of the values
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
    'outside': { x: 1.02, y: 1, xanchor: 'left', yanchor: 'top' }, // Legend outside plot area
    'top-right': { x: 1, y: 1, xanchor: 'right', yanchor: 'top' },
    'top-left': { x: 0, y: 1, xanchor: 'left', yanchor: 'top' },
    'bottom-right': { x: 1, y: 0, xanchor: 'right', yanchor: 'bottom' },
    'bottom-left': { x: 0, y: 0, xanchor: 'left', yanchor: 'bottom' },
  };

  const layout: Partial<PlotlyLayout> = useMemo(() => {
    const xRange = parseRange(figureSettings.xMin, figureSettings.xMax, figureSettings.logX);
    const yRange = parseRange(figureSettings.yMin, figureSettings.yMax, figureSettings.logY);
    
    // Convert inches to pixels (96 DPI standard for screens)
    const DPI = 96;
    const widthPx = Math.round(figureSettings.figWidthInches * DPI);
    const heightPx = Math.round(figureSettings.figHeightInches * DPI);
    
    // Calculate grid alpha based on settings
    const gridColor = `rgba(0,0,0,${figureSettings.gridAlpha})`;
    // Note: Plotly doesn't support minor grid separately, so we only use main grid
    
    // Don't set width in layout - let it be responsive
    // Publication-quality styling inspired by Matplotlib
    const layoutConfig = {
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
    
    console.log('[PlotViewer] Layout config:');
    console.log(JSON.stringify({
      xType: layoutConfig.xaxis.type,
      yType: layoutConfig.yaxis.type,
      xRange: xRange,
      yRange: yRange,
      xAutorange: !xRange,
      yAutorange: !yRange,
      logX: figureSettings.logX,
      logY: figureSettings.logY,
      widthInches: figureSettings.figWidthInches,
      heightInches: figureSettings.figHeightInches,
      widthPx,
      heightPx,
    }, null, 2));
    
    return layoutConfig;
  }, [figureSettings, legendPositionMap]);

  const config: Partial<PlotlyConfig> = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
  };

  const handleExport = async () => {
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
      link.download = `ace_plot.${exportOptions.format}`;
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
      setNotification({ type: 'error', message: 'Add at least one data series before saving' });
      return;
    }
    const name = configName.trim() || `ACE Plot - ${new Date().toLocaleString()}`;
    const snapshot: SavedPlotConfig = {
      id: createSeriesId(),
      name,
      createdAt: new Date().toISOString(),
      plotType,
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
    setNotification({ type: 'success', message: `Saved plot configuration "${name}"` });
  };

  const handleRestoreConfig = (configId: string) => {
    const target = savedConfigs.find((cfg) => cfg.id === configId);
    if (!target) return;
    const availableSeries = target.series.filter((series) =>
      files.some((file) => file.name === series.fileName),
    );
    if (availableSeries.length === 0) {
      setNotification({
        type: 'error',
        message: 'Unable to restore: required files are not loaded',
      });
      return;
    }
    updatePlotType(target.plotType, { preserveState: true });
    setFigureSettings({ ...target.figure });
    const restoredSeries = availableSeries.map((series) => ({
      ...series,
      id: createSeriesId(),
    }));
    setSeriesConfigs(restoredSeries);
    setSeriesDataMap({});
    const missing = target.series.length - availableSeries.length;
    const message =
      missing > 0
        ? `Restored "${target.name}" (skipped ${missing} missing file${missing > 1 ? 's' : ''})`
        : `Restored "${target.name}"`;
    setNotification({ type: 'success', message });
  };

  const handleDeleteConfig = (configId: string) => {
    setSavedConfigs((prev) => prev.filter((cfg) => cfg.id !== configId));
  };

  const prepareExportData = () => {
    return {
      series: seriesConfigs.map((series) => ({
        fileName: series.fileName,
        fileId: series.fileId,
        mtNumber: series.mtNumber,
        energy: series.energy,
        plotType: plotType,
        color: series.color,
        lineWidth: series.lineWidth,
        lineStyle: series.lineStyle,
        showMarkers: series.showMarkers,
        markerSymbol: String(series.markerSymbol || 'circle'),
        markerSize: series.markerSize,
        labelMode: series.labelMode,
        customLabel: series.customLabel,
      })),
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
    };
  };

  const handleGeneratePreview = async () => {
    if (seriesConfigs.length === 0) {
      setNotification({ type: 'error', message: 'No series to export' });
      return;
    }

    setGeneratingPreview(true);
    try {
      // Always generate PNG preview at lower DPI for quick display
      const previewData = {
        ...prepareExportData(),
        export_format: 'png',  // Always PNG for preview
        dpi: 150,  // Lower DPI for faster preview generation
      };
      const result = await exportWithMatplotlib(previewData);

      setPreviewImage({
        base64: result.image_base64,
        format: result.format,
        width: result.width_px,
        height: result.height_px,
        dpi: result.dpi,
      });
    } catch (error) {
      console.error('Preview generation error:', error);
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
      // Generate full quality export with user's chosen format and DPI
      const exportData = prepareExportData();
      const result = await exportWithMatplotlib(exportData);

      const link = document.createElement('a');
      link.href = `data:image/${result.format};base64,${result.image_base64}`;
      link.download = `ace_plot_hq.${result.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      setNotification({
        type: 'success',
        message: `High-quality plot downloaded (${result.width_px}×${result.height_px}px @ ${result.dpi} DPI)`,
      });
    } catch (error) {
      console.error('Export error:', error);
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
    if (showMatplotlibExport && !previewImage && !generatingPreview) {
      handleGeneratePreview();
    }
  }, [showMatplotlibExport]);

  // Regenerate preview when plot style changes
  useEffect(() => {
    if (showMatplotlibExport && !generatingPreview) {
      handleGeneratePreview();
    }
  }, [matplotlibExportSettings.style]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Alert severity="info" icon={<HighQualityIcon />}>
          <Typography variant="subtitle2" gutterBottom>
            <strong>Hybrid Plotting System</strong>
          </Typography>
          <Typography variant="body2">
            • <strong>Interactive Preview:</strong> Fast, real-time updates with Plotly (publication-styled)
            <br />
            • <strong>High-Quality Export:</strong> Use "Export (High Quality)" button for Matplotlib-rendered plots with full PlotBuilder customization
          </Typography>
        </Alert>
      </Grid>

      <Grid item xs={12} lg={4}>
        <Stack spacing={3}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Plot Workspace
            </Typography>
            <ToggleButtonGroup
              value={plotType}
              exclusive
              fullWidth
              onChange={(_, next) => {
                if (!next || next === plotType) return;
                updatePlotType(next);
              }}
              sx={{ mt: 2 }}
            >
              <ToggleButton value="xs">Cross Section</ToggleButton>
              <ToggleButton value="angular">Angular Distribution</ToggleButton>
            </ToggleButtonGroup>
            <Stack direction="row" spacing={1.5} sx={{ mt: 3 }}>
              <Button
                startIcon={<AddCircleOutlineIcon />}
                variant="contained"
                onClick={handleAddSeries}
                disabled={files.length === 0}
              >
                Add Series
              </Button>
              <Button
                startIcon={<DeleteOutlineIcon />}
                variant="outlined"
                onClick={handleClearSeries}
                disabled={seriesConfigs.length === 0}
              >
                Clear
              </Button>
            </Stack>
            {files.length === 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Upload ACE files on the Home tab to start plotting.
              </Alert>
            )}
          </Paper>

          {seriesConfigs.map((series, index) => {
            const state = seriesDataMap[series.id];
            const file = files.find((f) => f.name === series.fileName);
            const availableMts = getAvailableMts(file, plotType);
            return (
              <Accordion key={series.id} defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      size="small"
                      label={`Series ${index + 1}`}
                      color="primary"
                      variant="outlined"
                    />
                    <Typography variant="subtitle1">
                      {series.customLabel || series.autoLabel || state?.data?.label || 'Pending'}
                    </Typography>
                    {state?.status === 'loading' && <LinearProgress sx={{ width: 120 }} />}
                  </Box>
                </AccordionSummary>
                <AccordionDetails sx={{ p: 4 }}>
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
                      <InputLabel>ACE File</InputLabel>
                      <Select
                        value={series.fileName}
                        label="ACE File"
                        onChange={(event) => {
                          const selectedName = event.target.value;
                          const nextFile = files.find((f) => f.name === selectedName);
                          const mts = getAvailableMts(nextFile, plotType);
                          handleUpdateSeries(series.id, {
                            fileName: selectedName,
                            fileId: nextFile?.info?.file_id,
                            mtNumber: mts.length > 0 ? (mts.includes(2) ? 2 : mts[0]) : undefined,
                          });
                        }}
                      >
                        {files.map((fileOption) => (
                          <MenuItem key={fileOption.name} value={fileOption.name}>
                            {fileOption.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    <FormControl fullWidth>
                      <InputLabel>MT Number</InputLabel>
                      <Select
                        value={series.mtNumber ?? ''}
                        label="MT Number"
                        disabled={availableMts.length === 0}
                        onChange={(event) =>
                          handleUpdateSeries(series.id, {
                            mtNumber: Number(event.target.value),
                          })
                        }
                      >
                        {availableMts.map((mt) => (
                          <MenuItem key={mt} value={mt}>
                            MT {mt}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    {availableMts.length > 0 && (
                      <Typography variant="caption" color="text.secondary">
                        Available: {availableMts.slice(0, 12).join(', ')}
                        {availableMts.length > 12 ? ' ...' : ''}
                      </Typography>
                    )}

                    {plotType === 'angular' && (
                      <TextField
                        label="Energy (MeV)"
                        type="number"
                        value={series.energy ?? DEFAULT_ANGULAR_ENERGY}
                        onChange={(event) => {
                          const val = event.target.value;
                          const numVal = val === '' ? DEFAULT_ANGULAR_ENERGY : Number(val);
                          handleUpdateSeries(series.id, { energy: numVal });
                        }}
                        inputProps={{ 
                          min: 0.001, 
                          step: 0.1,
                          style: { MozAppearance: 'textfield' }
                        }}
                        sx={{
                          '& input[type=number]': {
                            MozAppearance: 'textfield',
                          },
                          '& input[type=number]::-webkit-outer-spin-button': {
                            WebkitAppearance: 'none',
                            margin: 0,
                          },
                          '& input[type=number]::-webkit-inner-spin-button': {
                            WebkitAppearance: 'none',
                            margin: 0,
                          },
                        }}
                        helperText="Energy at which to evaluate the angular distribution"
                      />
                    )}

                    <Divider />
                    <Typography variant="subtitle2">Label</Typography>
                    <ToggleButtonGroup
                      value={series.labelMode}
                      exclusive
                      onChange={(_, next) => {
                        if (!next) return;
                        handleUpdateSeries(series.id, { labelMode: next });
                      }}
                      size="small"
                    >
                      <ToggleButton value="auto">Auto</ToggleButton>
                      <ToggleButton value="custom">Custom</ToggleButton>
                    </ToggleButtonGroup>
                    {series.labelMode === 'custom' && (
                      <TextField
                        label="Custom Label"
                        value={series.customLabel}
                        onChange={(event) =>
                          handleUpdateSeries(series.id, { customLabel: event.target.value })
                        }
                      />
                    )}

                    <Divider sx={{ mt: 1 }} />
                    
                    {/* Collapsible Styling Section */}
                    <Accordion disableGutters elevation={0} sx={{ '&:before': { display: 'none' } }}>
                      <AccordionSummary 
                        expandIcon={<ExpandMoreIcon />}
                        sx={{ px: 0, minHeight: 40, '&.Mui-expanded': { minHeight: 40 } }}
                      >
                        <Typography variant="subtitle2">🎨 Styling Options</Typography>
                      </AccordionSummary>
                      <AccordionDetails sx={{ p: 4 }}>
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

                    {state?.status === 'error' && (
                      <Alert severity="error">{state.error ?? 'Failed to load data'}</Alert>
                    )}
                  </Stack>
                </AccordionDetails>
              </Accordion>
            );
          })}

          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Figure Settings</Typography>
              <Button
                size="small"
                startIcon={<RefreshIcon fontSize="small" />}
                onClick={() => setFigureSettings(getDefaultFigureSettings(plotType))}
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
                    {configInfo.plotType === 'xs' ? 'Cross Section' : 'Angular'} • {formatTimestamp(configInfo.createdAt)}
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
            <Alert severity={notification.type} onClose={() => setNotification(null)}>
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
                  👈 Configure series on the left and the plot will appear here in real time
                </Typography>
              </Box>
            ) : (
              <Box sx={{ 
                width: '100%',
                height: 'fit-content',
                maxWidth: '100%',
                overflow: 'auto',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
                bgcolor: 'background.paper',
              }}>
                <Plot
                  data={plotlyData}
                  layout={layout}
                  config={config}
                  style={{ width: '100%', display: 'block' }}
                  useResizeHandler={false}
                  divId="ace-plot"
                  onInitialized={(figure, graphDiv) => {
                    console.log('[PlotViewer] ✅ Plot initialized!', figure, graphDiv);
                    plotRef.current = graphDiv as PlotlyHTMLElement;
                  }}
                  onUpdate={(figure, graphDiv) => {
                    console.log('[PlotViewer] ✅ Plot updated!', figure);
                    plotRef.current = graphDiv as PlotlyHTMLElement;
                  }}
                  onError={(err) => {
                    console.error('[PlotViewer] ❌ Plot error:', err);
                  }}
                />
              </Box>
            )}

            <Divider />

            <Stack direction="row" spacing={2}>
              <Button
                variant="outlined"
                startIcon={<DownloadIcon />}
                onClick={handleExport}
                disabled={exporting || plotlyData.length === 0}
                fullWidth
              >
                {exporting ? 'Exporting...' : 'Quick Export (Plotly)'}
              </Button>
              <Button
                variant="contained"
                startIcon={<HighQualityIcon />}
                onClick={() => setShowMatplotlibExport(true)}
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

      {/* Matplotlib Export Dialog */}
      <Dialog 
        open={showMatplotlibExport} 
        onClose={() => {
          setShowMatplotlibExport(false);
          setPreviewImage(null);
        }} 
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Stack direction="row" alignItems="center" spacing={1}>
            <HighQualityIcon color="secondary" />
            <Typography variant="h6">High-Quality Export (Matplotlib)</Typography>
          </Stack>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 0 }}>
            {/* Left side: Settings */}
            <Grid item xs={12} md={6}>
              <Stack spacing={3}>
                <Alert severity="info" sx={{ fontSize: '0.875rem' }}>
                  Configure export settings and preview before downloading.
                </Alert>

                <FormControl fullWidth size="small">
                  <InputLabel>Plot Style</InputLabel>
                  <Select
                    value={matplotlibExportSettings.style}
                    label="Plot Style"
                    onChange={(e) => {
                      setMatplotlibExportSettings({ ...matplotlibExportSettings, style: e.target.value });
                      setPreviewImage(null);
                    }}
                  >
                    <MenuItem value="default">Default</MenuItem>
                    <MenuItem value="publication">Publication (Recommended)</MenuItem>
                    <MenuItem value="presentation">Presentation</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth size="small">
                  <InputLabel>Export Format</InputLabel>
                  <Select
                    value={matplotlibExportSettings.format}
                    label="Export Format"
                    onChange={(e) => {
                      setMatplotlibExportSettings({ ...matplotlibExportSettings, format: e.target.value });
                    }}
                  >
                    <MenuItem value="png">PNG (Raster)</MenuItem>
                    <MenuItem value="pdf">PDF (Vector, best for papers)</MenuItem>
                    <MenuItem value="svg">SVG (Vector, editable)</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth size="small">
                  <InputLabel>DPI (Resolution)</InputLabel>
                  <Select
                    value={matplotlibExportSettings.dpi}
                    label="DPI (Resolution)"
                    onChange={(e) => {
                      setMatplotlibExportSettings({ ...matplotlibExportSettings, dpi: Number(e.target.value) });
                    }}
                  >
                    <MenuItem value={150}>150 DPI (Draft)</MenuItem>
                    <MenuItem value={300}>300 DPI (Publication Standard)</MenuItem>
                    <MenuItem value={600}>600 DPI (High Quality)</MenuItem>
                    <MenuItem value={1200}>1200 DPI (Ultra High Quality)</MenuItem>
                  </Select>
                </FormControl>

                <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Export Settings
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Style: <strong>{matplotlibExportSettings.style}</strong>
                    <br />
                    Format: <strong>{matplotlibExportSettings.format.toUpperCase()}</strong>
                    <br />
                    Resolution: <strong>{matplotlibExportSettings.dpi} DPI</strong>
                    <br />
                    Size: <strong>
                      {figureSettings.figWidthInches}" × {figureSettings.figHeightInches}" 
                      ({Math.round(figureSettings.figWidthInches * matplotlibExportSettings.dpi)} × {Math.round(figureSettings.figHeightInches * matplotlibExportSettings.dpi)} px)
                    </strong>
                  </Typography>
                </Paper>
              </Stack>
            </Grid>

            {/* Right side: Preview */}
            <Grid item xs={12} md={6}>
              <Stack spacing={2}>
                <Typography variant="subtitle2">
                  Preview (PNG @ 150 DPI)
                </Typography>
                {generatingPreview ? (
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    justifyContent: 'center', 
                    alignItems: 'center',
                    minHeight: 300,
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 1,
                    bgcolor: 'grey.50',
                  }}>
                    <LinearProgress sx={{ width: '60%', mb: 2 }} />
                    <Typography variant="body2" color="text.secondary">
                      Generating preview...
                    </Typography>
                  </Box>
                ) : previewImage ? (
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'center', 
                      alignItems: 'center',
                      maxHeight: '400px',
                      overflow: 'auto',
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                      bgcolor: 'white',
                      p: 1,
                    }}>
                      <img 
                        src={`data:image/${previewImage.format};base64,${previewImage.base64}`}
                        alt="Plot preview"
                        style={{ maxWidth: '100%', height: 'auto', display: 'block' }}
                      />
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
                      Preview: {previewImage.width} × {previewImage.height} px
                    </Typography>
                  </Paper>
                ) : null}
                <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  Note: This is a quick preview. The downloaded file will be generated at the selected DPI and format.
                </Typography>
              </Stack>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => {
              setShowMatplotlibExport(false);
              setPreviewImage(null);
            }} 
            disabled={exporting}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleDownloadExport}
            disabled={exporting || generatingPreview}
            startIcon={exporting ? undefined : <DownloadIcon />}
          >
            {exporting ? 'Downloading...' : `Download ${matplotlibExportSettings.format.toUpperCase()}`}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};
