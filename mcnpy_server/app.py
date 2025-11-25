"""
MCNPy FastAPI Server
Simple local server for processing ACE files and generating plots
"""
import io
import base64
import hashlib
import time
import uuid
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tempfile

# Import MCNPy modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcnpy.ace.parsers import read_ace
from mcnpy.endf.read_endf import read_endf
from mcnpy.plotting.plot_builder import PlotBuilder
from mcnpy._utils import MeV_to_kelvin

app = FastAPI(title="MCNPy Processing Server", version="1.0.0")

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",  # Vite dev server
        "http://localhost:5173",  # Alternative Vite port
        "tauri://localhost",      # Tauri app
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACE_CACHE: Dict[str, Dict[str, Any]] = {}
ENDF_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_ITEMS = 16


class ACEParseRequest(BaseModel):
    """Request to parse ACE file content"""
    file_content: str
    file_name: str


class FileTypeDetectionResponse(BaseModel):
    """Response for file type auto-detection"""
    file_type: Optional[str]  # 'ace', 'endf', or None if unknown
    confidence: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None


class PlotRequest(BaseModel):
    """Request to generate plot from ACE data"""
    file_content: Optional[str] = None
    file_name: str
    plot_type: str  # 'xs' or 'angular'
    mt_number: Optional[int] = None  # Reaction MT number
    energy: Optional[float] = None  # Energy for angular distributions (MeV)
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None
    file_id: Optional[str] = None


class ACEInfoResponse(BaseModel):
    """Response with ACE file information"""
    file_id: str
    zaid: str
    atomic_weight_ratio: float
    temperature: float
    available_reactions: List[int]
    angular_reactions: List[int]
    has_angular_distributions: bool
    energy_grid_size: int


class PlotResponse(BaseModel):
    """Response with base64 encoded plot image"""
    image_base64: str
    format: str = "png"


class SeriesRequest(BaseModel):
    """Request to retrieve raw data arrays for a single series"""
    file_id: Optional[str] = None
    file_content: Optional[str] = None
    file_name: str
    plot_type: str  # 'xs' or 'angular'
    mt_number: int
    energy: Optional[float] = None


class SeriesResponse(BaseModel):
    """Response containing raw data for client-side plotting"""
    series_id: str
    label: str
    x: List[float]
    y: List[float]
    x_unit: str
    y_unit: str
    metadata: Dict[str, Any]
    suggested: Dict[str, Any]


class MatplotlibExportRequest(BaseModel):
    """Request to export plot using Matplotlib with full customization"""
    series: List[Dict[str, Any]]  # List of series configurations
    figure_settings: Dict[str, Any]  # Figure size, labels, scales, etc.
    style: str = "publication"  # 'default', 'publication', 'presentation', 'dark'
    export_format: str = "png"  # 'png', 'pdf', 'svg'
    dpi: int = 300


class MatplotlibExportResponse(BaseModel):
    """Response with high-quality Matplotlib export"""
    image_base64: str
    format: str
    width_px: int
    height_px: int
    dpi: int


class ENDFParseRequest(BaseModel):
    """Request to parse ENDF file content"""
    file_content: str
    file_name: str


class ENDFInfoResponse(BaseModel):
    """Response with ENDF file information"""
    file_id: str
    zaid: Optional[str] = None
    isotope: Optional[str] = None
    mat: Optional[int] = None
    has_mf4: bool = False
    has_mf34: bool = False
    angular_mts: List[int] = []
    uncertainty_mts: List[int] = []
    max_legendre_order_by_mt: Dict[str, int] = {}
    # New: explicit available Legendre orders per MT for MF4 and MF34
    available_orders_mf4_by_mt: Dict[str, List[int]] = {}
    available_orders_mf34_by_mt: Dict[str, List[int]] = {}


class ENDFSeriesRequest(BaseModel):
    """Request for ENDF plotting data"""
    file_id: Optional[str] = None
    file_content: Optional[str] = None
    file_name: str
    data_type: str  # 'angular' or 'uncertainty'
    mt_number: int
    order: int = 1
    include_uncertainty: bool = False
    sigma: float = 1.0


class ENDFUncertaintyPayload(BaseModel):
    """Uncertainty envelope for ENDF angular data"""
    lower: List[float]
    upper: List[float]
    sigma: float
    label: Optional[str] = None
    kind: str = "absolute"


class ENDFSeriesResponse(BaseModel):
    """Response containing ENDF plotting data"""
    series_id: str
    label: str
    x: List[float]
    y: List[float]
    x_unit: str
    y_unit: str
    plot_style: str = "line"
    line_shape: Optional[str] = None
    metadata: Dict[str, Any]
    suggested: Dict[str, Any]
    uncertainty: Optional[ENDFUncertaintyPayload] = None


class ENDFSeriesExportConfig(BaseModel):
    """Series configuration for ENDF Matplotlib export"""
    file_id: Optional[str] = None
    file_content: Optional[str] = None
    file_name: str
    data_type: str  # 'angular' or 'uncertainty'
    mt_number: int
    order: int
    include_uncertainty: bool = False
    sigma: float = 1.0
    color: Optional[str] = None
    lineWidth: Optional[float] = None
    lineStyle: Optional[str] = None
    showMarkers: bool = False
    markerSymbol: Optional[str] = None
    markerSize: Optional[float] = None
    labelMode: Optional[str] = None
    customLabel: Optional[str] = None


class ENDFMatplotlibExportRequest(BaseModel):
    """Matplotlib export request for ENDF data"""
    series: List[ENDFSeriesExportConfig]
    figure_settings: Dict[str, Any]
    style: str = "publication"
    export_format: str = "png"
    dpi: int = 300


def _generate_file_id(file_content: str) -> str:
    """Create a stable hash for the ACE file content."""
    return hashlib.sha256(file_content.encode('utf-8')).hexdigest()


def _cache_ace_object(file_id: str, file_name: str, ace_obj, info: Dict[str, Any]):
    """Store ACE object in memory with simple LRU eviction."""
    if len(ACE_CACHE) >= CACHE_MAX_ITEMS:
        oldest_key = min(ACE_CACHE.keys(), key=lambda key: ACE_CACHE[key]['timestamp'])
        ACE_CACHE.pop(oldest_key, None)
    ACE_CACHE[file_id] = {
        "ace": ace_obj,
        "file_name": file_name,
        "timestamp": time.time(),
        "info": info,
    }


def _get_cached_ace_entry(file_id: str):
    """Retrieve cached ACE object if available."""
    entry = ACE_CACHE.get(file_id)
    if entry:
        entry["timestamp"] = time.time()
    return entry


def _cache_endf_object(file_id: str, file_name: str, endf_obj, info: Dict[str, Any]):
    """Store ENDF object in memory with simple LRU eviction."""
    if len(ENDF_CACHE) >= CACHE_MAX_ITEMS:
        oldest_key = min(ENDF_CACHE.keys(), key=lambda key: ENDF_CACHE[key]['timestamp'])
        ENDF_CACHE.pop(oldest_key, None)
    ENDF_CACHE[file_id] = {
        "endf": endf_obj,
        "file_name": file_name,
        "timestamp": time.time(),
        "info": info,
    }


def _get_cached_endf_entry(file_id: str):
    """Retrieve cached ENDF object if available."""
    entry = ENDF_CACHE.get(file_id)
    if entry:
        entry["timestamp"] = time.time()
    return entry


def _is_valid_zaid(zaid) -> bool:
    """
    Check if ZAID is valid (not None, not 0, and reasonable value)
    ZAID format: ZZZAAA (Z = atomic number, A = mass number)
    """
    if zaid is None:
        return False
    
    try:
        zaid_int = int(zaid)
        # ZAID should be positive and reasonable (1000 to 999999)
        if zaid_int <= 0 or zaid_int > 999999:
            return False
        
        # Extract Z (atomic number) and A (mass number)
        z = zaid_int // 1000
        a = zaid_int % 1000
        
        # Basic validation: Z should be 1-118 (known elements)
        if z < 1 or z > 118:
            return False
        
        # A should be reasonable (0-400, including metastable states)
        if a > 400:
            return False
        
        return True
    except (ValueError, TypeError):
        return False


def _detect_file_type(file_content: str, file_name: str) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Auto-detect file type by trying to parse it
    
    Strategy:
    1. Try reading as ENDF first
    2. Check if ZAID is valid - if yes, it's ENDF
    3. If not valid, try reading as ACE
    4. Check if ZAID is valid - if yes, it's ACE
    5. If neither works, return None
    
    Returns
    -------
    Tuple[Optional[str], float, Dict[str, Any]]
        (file_type, confidence, details)
        file_type: 'endf', 'ace', or None
        confidence: 0.0 to 1.0
        details: Additional metadata from parsing attempt
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False, encoding='utf-8') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Try ENDF first
        try:
            endf_obj = read_endf(tmp_path)
            if endf_obj:
                # Check for valid ZAID
                zaid = getattr(endf_obj, 'zaid', None)
                if _is_valid_zaid(zaid):
                    # Additional validation: should have MAT and MF
                    if hasattr(endf_obj, 'mat') and hasattr(endf_obj, 'mf'):
                        return 'endf', 0.95, {
                            'zaid': str(zaid),
                            'isotope': getattr(endf_obj, 'isotope', 'unknown'),
                            'mat': getattr(endf_obj, 'mat', None),
                        }
        except Exception as endf_error:
            pass
        
        # Try ACE if ENDF failed or had invalid ZAID
        try:
            ace_obj = read_ace(tmp_path, debug=False)
            if ace_obj:
                # Check for valid ZAID
                zaid = getattr(ace_obj, 'zaid', None)
                if _is_valid_zaid(zaid):
                    return 'ace', 0.95, {
                        'zaid': str(zaid),
                        'temperature': float(getattr(getattr(ace_obj, 'header', None), 'temperature', 0.0) or 0.0),
                    }
        except Exception as ace_error:
            pass
        
        # Neither format worked
        return None, 0.0, {'error': 'Could not parse as ENDF or ACE'}
    
    finally:
        if tmp_path:
            path_obj = Path(tmp_path)
            if path_obj.exists():
                path_obj.unlink()


def _parse_ace_content(file_content: str, file_name: str):
    """Parse ACE content string into an ACE object."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ace', delete=False, encoding='utf-8') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        ace_obj = read_ace(tmp_path, debug=True)
        return ace_obj
    finally:
        if tmp_path:
            path_obj = Path(tmp_path)
            if path_obj.exists():
                path_obj.unlink()


def _extract_ace_metadata(ace_obj) -> Dict[str, Any]:
    """Convert ACE object properties into serializable metadata."""
    available_raw = getattr(ace_obj, 'mt_numbers', None)
    if available_raw is None:
        available_reactions = []
    else:
        available_reactions = sorted({int(mt) for mt in available_raw})

    angular_reactions: List[int] = []
    has_angular = False
    angular_data = getattr(ace_obj, 'angular_distributions', None)
    if angular_data:
        if hasattr(angular_data, 'get_neutron_reaction_mt_numbers'):
            angular_reactions = angular_data.get_neutron_reaction_mt_numbers() or []
        if getattr(angular_data, 'has_elastic_data', False) and 2 not in angular_reactions:
            angular_reactions = [2] + angular_reactions
        angular_reactions = sorted(set(angular_reactions))
        has_angular = len(angular_reactions) > 0
    temp_kelvin = 0.0
    if getattr(ace_obj, 'header', None) and getattr(ace_obj.header, 'temperature', None):
        temp_mev = float(ace_obj.header.temperature)
        temp_kelvin = MeV_to_kelvin(temp_mev)

    energies = getattr(ace_obj, 'energies', None)
    energy_grid_size = len(energies) if energies is not None else 0

    return {
        "zaid": str(getattr(ace_obj, 'zaid', 'unknown')),
        "atomic_weight_ratio": float(getattr(getattr(ace_obj, 'header', None), 'atomic_weight_ratio', 0.0) or 0.0),
        "temperature": temp_kelvin,
        "available_reactions": available_reactions,
        "angular_reactions": angular_reactions,
        "has_angular_distributions": has_angular,
        "energy_grid_size": energy_grid_size,
    }


def _load_ace_object(
    file_id: Optional[str],
    file_content: Optional[str],
    file_name: str,
) -> Tuple[str, Any, Dict[str, Any]]:
    """Load ACE object either from cache or by parsing new content."""
    if file_id:
        cached = _get_cached_ace_entry(file_id)
        if cached:
            return file_id, cached["ace"], cached.get("info", {})
        # If file_id provided but not cached, fall back to file_content if available.
        if not file_content:
            raise HTTPException(status_code=404, detail="ACE file not found in cache. Please provide file_content.")
    if not file_content:
        raise HTTPException(status_code=400, detail="Either file_id or file_content must be provided.")

    derived_id = _generate_file_id(file_content)
    cached = _get_cached_ace_entry(derived_id)
    if cached:
        return derived_id, cached["ace"], cached.get("info", {})

    ace_obj = _parse_ace_content(file_content, file_name)
    info = _extract_ace_metadata(ace_obj)
    _cache_ace_object(derived_id, file_name, ace_obj, info)
    return derived_id, ace_obj, info


def _parse_endf_content(file_content: str, file_name: str):
    """Parse ENDF content string into an ENDF object."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.endf', delete=False, encoding='utf-8') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        endf_obj = read_endf(tmp_path)
        return endf_obj
    finally:
        if tmp_path:
            path_obj = Path(tmp_path)
            if path_obj.exists():
                path_obj.unlink()


def _extract_endf_metadata(endf_obj) -> Dict[str, Any]:
    """Convert ENDF object metadata into serializable information."""
    has_mf4 = 4 in getattr(endf_obj, 'mf', {})
    has_mf34 = 34 in getattr(endf_obj, 'mf', {})
    angular_mts: List[int] = []
    uncertainty_mts: List[int] = []
    order_map: Dict[str, int] = {}
    orders_mf4_map: Dict[str, List[int]] = {}
    orders_mf34_map: Dict[str, List[int]] = {}

    if has_mf4:
        mf4 = endf_obj.mf[4]
        angular_mts = sorted(int(mt) for mt in mf4.mt.keys())
        for mt, section in mf4.mt.items():
            max_order = 0
            coeffs = getattr(section, 'legendre_coefficients', None)
            if coeffs:
                lengths = [len(row) for row in coeffs if row]
                if lengths:
                    max_order = max(lengths) - 1  # Subtract 1 because length includes L=0
            if max_order == 0 and hasattr(section, 'num_legendre_coefficients'):
                try:
                    max_order = int(getattr(section, 'num_legendre_coefficients') or 0) - 1
                except Exception:
                    max_order = 0
            order_map[str(int(mt))] = int(max_order)
            # Available orders list for MF4: include L=0 always, then 1..max
            try:
                available = list(range(max(0, int(max_order)) + 1))
            except Exception:
                available = [0]
            orders_mf4_map[str(int(mt))] = available
    if has_mf34:
        mf34 = endf_obj.mf[34]
        uncertainty_mts = sorted(int(mt) for mt in mf34.mt.keys())
        # Extract max order for MF34 (uncertainty data)
        # Determine available orders using MF34 covariance matrices for the specific isotope/MT
        try:
            # Convert MF34 to a combined covariance matrix object
            mf34_obj = mf34
            ang_covmat = mf34_obj.to_ang_covmat() if hasattr(mf34_obj, 'to_ang_covmat') else None
        except Exception:
            ang_covmat = None

        for mt, section in mf34.mt.items():
            mt_key = str(int(mt))
            max_order = 0
            # Default: try to infer max order from section metadata when available
            if hasattr(section, 'legendre_coefficients'):
                coeffs = getattr(section, 'legendre_coefficients', None)
                if coeffs:
                    lengths = [len(row) for row in coeffs if row]
                    if lengths:
                        max_order = max(lengths) - 1
            elif hasattr(section, 'num_legendre_coefficients'):
                try:
                    max_order = int(getattr(section, 'num_legendre_coefficients') or 0) - 1
                except Exception:
                    max_order = 0

            # If covariance object available, compute precise set of L values for this isotope+MT
            available_ls: List[int] = []
            try:
                zaid = getattr(endf_obj, 'zaid', None)
                zaid_int = int(zaid) if zaid is not None else None
                if ang_covmat is not None and zaid_int is not None:
                    filtered = ang_covmat.filter_by_isotope_reaction(zaid_int, int(mt))
                    # legendre_indices property returns a sorted set of available L values
                    available_ls = list(getattr(filtered, 'legendre_indices', []) or [])
            except Exception:
                # Fall back to contiguous range if filtering fails
                available_ls = []

            if available_ls:
                # Ensure L=0 is present (uncertainty of a0 may or may not be in MF34; include if present)
                orders_mf34_map[mt_key] = sorted(set(available_ls))
                max_order = max(max_order, max(available_ls))
            else:
                # Fallback to contiguous range if we couldn't get indices
                orders_mf34_map[mt_key] = list(range(max(0, int(max_order)) + 1))

            # Populate order_map if not already set (keep MF4-derived max if available)
            if mt_key not in order_map:
                order_map[mt_key] = int(max_order)

    zaid = getattr(endf_obj, 'zaid', None)
    return {
        "zaid": str(zaid) if zaid is not None else None,
        "isotope": getattr(endf_obj, 'isotope', None),
        "mat": getattr(endf_obj, 'mat', None),
        "has_mf4": has_mf4,
        "has_mf34": has_mf34,
        "angular_mts": angular_mts,
        "uncertainty_mts": uncertainty_mts,
        "max_legendre_order_by_mt": order_map,
        "available_orders_mf4_by_mt": orders_mf4_map,
        "available_orders_mf34_by_mt": orders_mf34_map,
    }


def _load_endf_object(
    file_id: Optional[str],
    file_content: Optional[str],
    file_name: str,
) -> Tuple[str, Any, Dict[str, Any]]:
    """Load ENDF object either from cache or by parsing new content."""
    if file_id:
        cached = _get_cached_endf_entry(file_id)
        if cached:
            return file_id, cached["endf"], cached.get("info", {})
        if not file_content:
            raise HTTPException(status_code=404, detail="ENDF file not found in cache. Please provide file_content.")
    if not file_content:
        raise HTTPException(status_code=400, detail="Either file_id or file_content must be provided.")

    derived_id = _generate_file_id(file_content)
    cached = _get_cached_endf_entry(derived_id)
    if cached:
        return derived_id, cached["endf"], cached.get("info", {})

    endf_obj = _parse_endf_content(file_content, file_name)
    info = _extract_endf_metadata(endf_obj)
    _cache_endf_object(derived_id, file_name, endf_obj, info)
    return derived_id, endf_obj, info


@app.get("/")
async def root():
    return {
        "name": "MCNPy Processing Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/detect-file-type", response_model=FileTypeDetectionResponse)
async def detect_file_type(request: ACEParseRequest):
    """
    Auto-detect file type by attempting to parse
    
    Strategy:
    1. Try to parse as ENDF
    2. Try to parse as ACE
    3. Return the detected type with confidence score
    """
    try:
        file_type, confidence, details = _detect_file_type(request.file_content, request.file_name)
        return FileTypeDetectionResponse(
            file_type=file_type,
            confidence=confidence,
            details=details
        )
    except Exception as e:
        return FileTypeDetectionResponse(
            file_type=None,
            confidence=0.0,
            details={"error": str(e)}
        )


@app.post("/api/ace/parse", response_model=ACEInfoResponse)
async def parse_ace_file(request: ACEParseRequest):
    """
    Parse ACE file and return basic information
    """
    try:
        file_id, ace_obj, info = _load_ace_object(
            file_id=None,
            file_content=request.file_content,
            file_name=request.file_name,
        )
        # Ensure cache entry records latest file name
        entry = ACE_CACHE.get(file_id)
        if entry:
            entry["file_name"] = request.file_name
        if not info:
            info = _extract_ace_metadata(ace_obj)
            _cache_ace_object(file_id, request.file_name, ace_obj, info)
        return ACEInfoResponse(file_id=file_id, **info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing ACE file: {str(e)}")


@app.post("/api/endf/parse", response_model=ENDFInfoResponse)
async def parse_endf_file(request: ENDFParseRequest):
    """
    Parse ENDF file and return metadata needed by the frontend
    """
    try:
        file_id, endf_obj, info = _load_endf_object(
            file_id=None,
            file_content=request.file_content,
            file_name=request.file_name,
        )
        entry = ENDF_CACHE.get(file_id)
        if entry:
            entry["file_name"] = request.file_name
        if not info:
            info = _extract_endf_metadata(endf_obj)
            _cache_endf_object(file_id, request.file_name, endf_obj, info)
        return ENDFInfoResponse(
            file_id=file_id,
            **info,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing ENDF file: {str(e)}")


@app.post("/api/ace/plot", response_model=PlotResponse)
async def generate_plot(request: PlotRequest):
    """
    Generate plot from ACE file data and return as base64 image
    """
    try:
        file_id, ace_obj, _ = _load_ace_object(
            file_id=request.file_id,
            file_content=request.file_content,
            file_name=request.file_name,
        )
        
        # Create plot based on type using to_plot_data and PlotBuilder
        fig = None
        
        if request.plot_type == 'xs' and request.mt_number:
            # Cross section plot using to_plot_data
            plot_data = ace_obj.to_plot_data('xs', mt=request.mt_number)
            
            builder = PlotBuilder(figsize=(10, 6))
            builder.add_data(plot_data)
            builder.set_labels(
                title=f'{ace_obj.zaid} - MT{request.mt_number}',
                x_label='Energy (MeV)',
                y_label='Cross Section (barns)'
            )
            builder.set_scales(log_x=True, log_y=True)
            fig = builder.build()
                
        elif request.plot_type == 'angular' and request.mt_number:
            # Angular distribution plot using to_plot_data
            # For angular, we need an energy value from request or use default
            energy = request.energy if request.energy is not None else 1.0  # Default 1 MeV
            
            plot_data = ace_obj.to_plot_data('angular', mt=request.mt_number, energy=energy)
            
            builder = PlotBuilder(figsize=(10, 6))
            builder.add_data(plot_data)
            builder.set_labels(
                title=f'{ace_obj.zaid} - Angular Distribution MT{request.mt_number} at {energy} MeV',
                x_label='Cosine of Scattering Angle',
                y_label='Probability'
            )
            fig = builder.build()
        
        if fig is None:
            raise HTTPException(status_code=400, detail="Could not generate plot - no data available")
        
        # Convert figure to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return PlotResponse(image_base64=image_base64)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")


@app.post("/api/ace/series", response_model=SeriesResponse)
async def get_series_data(request: SeriesRequest):
    """
    Return raw x/y data so the client can render interactive plots.
    """
    try:
        file_id, ace_obj, info = _load_ace_object(
            file_id=request.file_id,
            file_content=request.file_content,
            file_name=request.file_name,
        )
        if not info:
            info = _extract_ace_metadata(ace_obj)

        if request.plot_type == 'xs':
            plot_data = ace_obj.to_plot_data('xs', mt=request.mt_number)
            x_unit = "Energy (MeV)"
            y_unit = "Cross Section (barns)"
            suggested = {
                "x_label": "Energy (MeV)",
                "y_label": "Cross Section (barns)",
                "log_x": True,
                "log_y": True,
                "title": f"{info.get('zaid', 'ACE')} MT{request.mt_number}",
            }
            energy_value = None
        else:
            energy_value = request.energy if request.energy is not None else 1.0
            plot_data = ace_obj.to_plot_data('angular', mt=request.mt_number, energy=energy_value)
            x_unit = "cos(theta)"
            y_unit = "Probability Density"
            suggested = {
                "x_label": "cos(theta)",
                "y_label": "Probability Density",
                "log_x": False,
                "log_y": False,
                "title": f"{info.get('zaid', 'ACE')} MT{request.mt_number} @ {energy_value} MeV",
            }

        metadata = {
            "file_id": file_id,
            "file_name": request.file_name,
            "mt_number": request.mt_number,
            "plot_type": request.plot_type,
            "energy": energy_value,
            "zaid": info.get('zaid', 'unknown'),
            "point_count": len(plot_data.x),
        }

        return SeriesResponse(
            series_id=uuid.uuid4().hex,
            label=plot_data.label or suggested["title"],
            x=plot_data.x.tolist(),
            y=plot_data.y.tolist(),
            x_unit=x_unit,
            y_unit=y_unit,
            metadata=metadata,
            suggested=suggested,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting data series: {str(e)}")


@app.post("/api/endf/series", response_model=ENDFSeriesResponse)
async def get_endf_series(request: ENDFSeriesRequest):
    """
    Return ENDF Legendre/uncertainty data for interactive plotting
    """
    try:
        file_id, endf_obj, info = _load_endf_object(
            file_id=request.file_id,
            file_content=request.file_content,
            file_name=request.file_name,
        )
        data_type = request.data_type.lower()
        if data_type not in ('angular', 'uncertainty'):
            raise HTTPException(status_code=400, detail="data_type must be 'angular' or 'uncertainty'")

        sigma = request.sigma if request.sigma and request.sigma > 0 else 1.0
        include_unc = bool(request.include_uncertainty)

        if data_type == 'angular':
            result = endf_obj.to_plot_data(
                mf=4,
                mt=request.mt_number,
                order=request.order,
                uncertainty=include_unc,
                sigma=sigma,
            )
            if isinstance(result, tuple):
                plot_data, unc_band = result
            else:
                plot_data = result
                unc_band = None
            x_unit = "Energy (eV)"
            y_unit = "Legendre Coefficient"
        else:
            plot_data = endf_obj.to_plot_data(
                mf=34,
                mt=request.mt_number,
                order=request.order,
            )
            unc_band = None
            x_unit = "Energy (eV)"
            y_unit = "Relative Uncertainty"

        label = plot_data.label or f"{info.get('isotope', 'ENDF')} MT{request.mt_number} L={request.order}"
        plot_style = getattr(plot_data, 'plot_type', 'line') or 'line'
        line_shape = 'hv' if plot_style == 'step' else None

        uncertainty_payload = None
        if unc_band:
            try:
                # Handle LegendreUncertaintyPlotData which has x, y attributes (uncertainty on sparse grid)
                # Check if it's the new style object (has x, y, uncertainty_type)
                if hasattr(unc_band, 'uncertainty_type') and hasattr(unc_band, 'x') and hasattr(unc_band, 'y'):
                    from scipy.interpolate import interp1d
                    
                    # Interpolate uncertainty to match nominal data energy grid
                    # Use 'previous' (step) interpolation to match the native step plot behavior
                    interp_func = interp1d(
                        unc_band.x, 
                        unc_band.y, 
                        kind='previous', 
                        bounds_error=False, 
                        fill_value=(unc_band.y[0], unc_band.y[-1])
                    )
                    unc_interp = interp_func(plot_data.x)
                    
                    # Convert from relative (%) to absolute bounds
                    y_arr = np.array(plot_data.y)
                    if unc_band.uncertainty_type == 'relative':
                        unc_relative = np.array(unc_interp) / 100.0  # Convert % to fraction
                        lower_arr = y_arr - y_arr * unc_relative
                        upper_arr = y_arr + y_arr * unc_relative
                    else:
                        # Absolute uncertainty (rare case)
                        lower_arr = y_arr - unc_interp
                        upper_arr = y_arr + unc_interp
                        
                # Old style with y_lower/y_upper attributes (backwards compatibility)
                elif hasattr(unc_band, 'y_lower') and hasattr(unc_band, 'y_upper'):
                    is_relative = False
                    if hasattr(unc_band, 'is_relative') and callable(unc_band.is_relative):
                        is_relative = unc_band.is_relative()
                    elif hasattr(unc_band, '_is_relative'):
                        is_relative = unc_band._is_relative
                    
                    if is_relative:
                        y_arr = np.array(plot_data.y)
                        lower_arr = y_arr - y_arr * np.array(unc_band.y_lower)
                        upper_arr = y_arr + y_arr * np.array(unc_band.y_upper)
                    else:
                        lower_arr = unc_band.y_lower
                        upper_arr = unc_band.y_upper
                else:
                    raise ValueError(f"Uncertainty band object has unexpected structure. Type: {type(unc_band)}, Attributes: {dir(unc_band)}")
                
                if lower_arr is not None and upper_arr is not None:
                    lower_list = lower_arr.tolist() if hasattr(lower_arr, 'tolist') else list(lower_arr)
                    upper_list = upper_arr.tolist() if hasattr(upper_arr, 'tolist') else list(upper_arr)
                    uncertainty_payload = ENDFUncertaintyPayload(
                        lower=lower_list,
                        upper=upper_list,
                        sigma=float(getattr(unc_band, 'sigma', sigma)),
                        label=getattr(unc_band, 'label', None),
                        kind="absolute",
                    )
            except Exception as unc_error:
                # Log but don't fail the entire request if uncertainty processing fails
                print(f"Warning: Could not process uncertainty band: {str(unc_error)}")
                uncertainty_payload = None

        metadata = {
            "file_id": file_id,
            "file_name": request.file_name,
            "data_type": data_type,
            "mt_number": request.mt_number,
            "order": request.order,
            "include_uncertainty": include_unc,
            "sigma": sigma,
            "zaid": info.get('zaid'),
        }

        suggested = {
            "x_label": "Energy (eV)",
            "y_label": "Legendre Coefficient" if data_type == 'angular' else "Relative Uncertainty",
            "title": f"{info.get('isotope', 'ENDF')} MT{request.mt_number} L={request.order}",
            "log_x": True,
            "log_y": False,
        }

        return ENDFSeriesResponse(
            series_id=uuid.uuid4().hex,
            label=label,
            x=plot_data.x.tolist(),
            y=plot_data.y.tolist(),
            x_unit=x_unit,
            y_unit=y_unit,
            plot_style=plot_style,
            line_shape=line_shape,
            metadata=metadata,
            suggested=suggested,
            uncertainty=uncertainty_payload,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting ENDF data series: {str(e)}")


@app.post("/api/ace/upload")
async def upload_ace_file(file: UploadFile = File(...)):
    """
    Upload and parse ACE file (alternative endpoint using multipart/form-data)
    """
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        request = ACEParseRequest(
            file_content=content_str,
            file_name=file.filename or "unknown.ace"
        )
        
        return await parse_ace_file(request)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded file: {str(e)}")


@app.post("/api/plot/matplotlib-export", response_model=MatplotlibExportResponse)
async def export_with_matplotlib(request: MatplotlibExportRequest):
    """
    Generate high-quality plot using Matplotlib PlotBuilder with full customization.
    This endpoint is designed for final exports (publications, presentations).
    """
    try:
        fig_settings = request.figure_settings
        
        # Extract figure size
        fig_width = fig_settings.get('width', 900) / 100  # Convert px to inches (assuming 100 dpi)
        fig_height = fig_settings.get('height', 580) / 100
        
        # Create PlotBuilder with style
        builder = PlotBuilder(
            style=request.style,
            figsize=(fig_width, fig_height),
            dpi=request.dpi
        )
        
        # Process each series
        for series_config in request.series:
            file_id = series_config.get('fileId')
            file_name = series_config.get('fileName', 'unknown')
            mt_number = series_config.get('mtNumber')
            plot_type = series_config.get('plotType', 'xs')
            
            if not file_id or mt_number is None:
                continue
                
            # Load ACE object
            _, ace_obj, _ = _load_ace_object(
                file_id=file_id,
                file_content=None,
                file_name=file_name
            )
            
            # Generate plot data
            if plot_type == 'xs':
                plot_data = ace_obj.to_plot_data('xs', mt=mt_number)
            else:  # angular
                energy = series_config.get('energy', 1.0)
                plot_data = ace_obj.to_plot_data('angular', mt=mt_number, energy=energy)
            
            # Apply series styling
            plot_data.color = series_config.get('color', None)
            plot_data.linewidth = series_config.get('lineWidth', 2.0)
            plot_data.linestyle = _convert_plotly_linestyle(series_config.get('lineStyle', 'solid'))
            
            # Handle markers
            if series_config.get('showMarkers', False):
                plot_data.marker = _convert_plotly_marker(series_config.get('markerSymbol', 'circle'))
                plot_data.markersize = series_config.get('markerSize', 6)
            
            # Override label if custom
            if series_config.get('labelMode') == 'custom' and series_config.get('customLabel'):
                plot_data.label = series_config['customLabel']
            
            builder.add_data(plot_data)
        
        # Configure plot
        builder.set_labels(
            title=fig_settings.get('title', ''),
            x_label=fig_settings.get('xLabel', ''),
            y_label=fig_settings.get('yLabel', '')
        )
        
        builder.set_scales(
            log_x=fig_settings.get('logX', False),
            log_y=fig_settings.get('logY', False)
        )
        
        # Set axis limits if specified
        x_min = fig_settings.get('xMin')
        x_max = fig_settings.get('xMax')
        y_min = fig_settings.get('yMin')
        y_max = fig_settings.get('yMax')
        
        x_lim = None
        y_lim = None
        
        if x_min or x_max:
            try:
                x_min_val = float(x_min) if x_min else None
                x_max_val = float(x_max) if x_max else None
                if x_min_val is not None or x_max_val is not None:
                    x_lim = (x_min_val, x_max_val)
            except (ValueError, TypeError):
                pass
        
        if y_min or y_max:
            try:
                y_min_val = float(y_min) if y_min else None
                y_max_val = float(y_max) if y_max else None
                if y_min_val is not None or y_max_val is not None:
                    y_lim = (y_min_val, y_max_val)
            except (ValueError, TypeError):
                pass
        
        if x_lim is not None or y_lim is not None:
            builder.set_limits(x_lim=x_lim, y_lim=y_lim)
        
        # Grid and legend
        if fig_settings.get('showGrid', True):
            grid_alpha = fig_settings.get('gridAlpha', 0.3)
            show_minor = fig_settings.get('showMinorGrid', False)
            minor_alpha = fig_settings.get('minorGridAlpha', 0.15)
            builder.set_grid(grid=True, alpha=grid_alpha, show_minor=show_minor, minor_alpha=minor_alpha)
        else:
            builder.set_grid(grid=False)
        
        if fig_settings.get('showLegend', True):
            # Convert Plotly legend position to matplotlib
            legend_pos = fig_settings.get('legendPosition', 'top-right')
            mpl_legend_pos = {
                'top-right': 'upper right',
                'top-left': 'upper left',
                'bottom-right': 'lower right',
                'bottom-left': 'lower left',
                'outside': 'upper left'  # Outside position - we'll adjust in build
            }.get(legend_pos, 'best')
            builder.set_legend(loc=mpl_legend_pos)
        
        # Build the figure
        fig = builder.build()
        
        # Export to specified format
        buf = io.BytesIO()
        fig.savefig(buf, format=request.export_format, dpi=request.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Calculate actual dimensions
        width_px = int(fig_width * request.dpi)
        height_px = int(fig_height * request.dpi)
        
        return MatplotlibExportResponse(
            image_base64=image_base64,
            format=request.export_format,
            width_px=width_px,
            height_px=height_px,
            dpi=request.dpi
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating matplotlib export: {str(e)}")


@app.post("/api/plot/endf/matplotlib-export", response_model=MatplotlibExportResponse)
async def export_endf_with_matplotlib(request: ENDFMatplotlibExportRequest):
    """
    Generate high-quality ENDF plot using Matplotlib with full customization.
    """
    if not request.series:
        raise HTTPException(status_code=400, detail="No series provided for export")

    try:
        fig_settings = request.figure_settings
        fig_width = fig_settings.get('width', 900) / 100
        fig_height = fig_settings.get('height', 580) / 100

        builder = PlotBuilder(
            style=request.style,
            figsize=(fig_width, fig_height),
            dpi=request.dpi
        )

        for idx, series_config in enumerate(request.series):
            if not series_config.file_id and not series_config.file_content:
                raise HTTPException(status_code=400, detail=f"Series {idx+1} missing file_id and file_content")

            _, endf_obj, _ = _load_endf_object(
                file_id=series_config.file_id,
                file_content=series_config.file_content,
                file_name=series_config.file_name,
            )

            data_type = series_config.data_type.lower()
            sigma = series_config.sigma if series_config.sigma and series_config.sigma > 0 else 1.0

            if data_type == 'angular':
                result = endf_obj.to_plot_data(
                    mf=4,
                    mt=series_config.mt_number,
                    order=series_config.order,
                    uncertainty=series_config.include_uncertainty,
                    sigma=sigma,
                )
                if isinstance(result, tuple):
                    plot_data, unc_band = result
                else:
                    plot_data = result
                    unc_band = None
            elif data_type == 'uncertainty':
                plot_data = endf_obj.to_plot_data(
                    mf=34,
                    mt=series_config.mt_number,
                    order=series_config.order,
                )
                unc_band = None
            else:
                raise HTTPException(status_code=400, detail=f"Invalid data_type '{series_config.data_type}' in series {idx+1}")

            # Styling
            if series_config.labelMode == 'custom' and series_config.customLabel:
                plot_data.label = series_config.customLabel

            if series_config.color:
                plot_data.color = series_config.color
            if series_config.lineWidth is not None:
                plot_data.linewidth = series_config.lineWidth
            if series_config.lineStyle:
                plot_data.linestyle = _convert_plotly_linestyle(series_config.lineStyle)

            if series_config.showMarkers and series_config.markerSymbol:
                plot_data.marker = _convert_plotly_marker(series_config.markerSymbol)
                if series_config.markerSize is not None:
                    plot_data.markersize = series_config.markerSize
            else:
                plot_data.marker = None

            if unc_band and not series_config.include_uncertainty:
                unc_band = None

            if unc_band:
                builder.add_data(plot_data, uncertainty=unc_band)
            else:
                builder.add_data(plot_data)

        # Configure figure settings (same as ACE export)
        builder.set_labels(
            title=fig_settings.get('title', ''),
            x_label=fig_settings.get('xLabel', ''),
            y_label=fig_settings.get('yLabel', '')
        )

        builder.set_scales(
            log_x=fig_settings.get('logX', False),
            log_y=fig_settings.get('logY', False)
        )

        x_lim = None
        y_lim = None

        x_min = fig_settings.get('xMin')
        x_max = fig_settings.get('xMax')
        y_min = fig_settings.get('yMin')
        y_max = fig_settings.get('yMax')

        if x_min or x_max:
            try:
                x_min_val = float(x_min) if x_min else None
                x_max_val = float(x_max) if x_max else None
                if x_min_val is not None or x_max_val is not None:
                    x_lim = (x_min_val, x_max_val)
            except (ValueError, TypeError):
                pass

        if y_min or y_max:
            try:
                y_min_val = float(y_min) if y_min else None
                y_max_val = float(y_max) if y_max else None
                if y_min_val is not None or y_max_val is not None:
                    y_lim = (y_min_val, y_max_val)
            except (ValueError, TypeError):
                pass

        if x_lim is not None or y_lim is not None:
            builder.set_limits(x_lim=x_lim, y_lim=y_lim)

        if fig_settings.get('showGrid', True):
            grid_alpha = fig_settings.get('gridAlpha', 0.3)
            show_minor = fig_settings.get('showMinorGrid', False)
            minor_alpha = fig_settings.get('minorGridAlpha', 0.15)
            builder.set_grid(grid=True, alpha=grid_alpha, show_minor=show_minor, minor_alpha=minor_alpha)
        else:
            builder.set_grid(grid=False)

        if fig_settings.get('showLegend', True):
            legend_pos = fig_settings.get('legendPosition', 'top-right')
            mpl_legend_pos = {
                'top-right': 'upper right',
                'top-left': 'upper left',
                'bottom-right': 'lower right',
                'bottom-left': 'lower left',
                'outside': 'upper left',
            }.get(legend_pos, 'best')
            builder.set_legend(loc=mpl_legend_pos)

        fig = builder.build()

        buf = io.BytesIO()
        fig.savefig(buf, format=request.export_format, dpi=request.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        width_px = int(fig_width * request.dpi)
        height_px = int(fig_height * request.dpi)

        return MatplotlibExportResponse(
            image_base64=image_base64,
            format=request.export_format,
            width_px=width_px,
            height_px=height_px,
            dpi=request.dpi
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating ENDF matplotlib export: {str(e)}")


def _convert_plotly_linestyle(plotly_style: str) -> str:
    """Convert Plotly line style to Matplotlib line style"""
    mapping = {
        'solid': '-',
        'dash': '--',
        'dot': ':',
        'dashdot': '-.',
    }
    return mapping.get(plotly_style, '-')


def _convert_plotly_marker(plotly_marker: str) -> str:
    """Convert Plotly marker symbol to Matplotlib marker"""
    mapping = {
        'circle': 'o',
        'square': 's',
        'triangle-up': '^',
        'triangle-down': 'v',
        'diamond': 'D',
        'star': '*',
        'x': 'x',
        'cross': '+',
    }
    return mapping.get(plotly_marker, 'o')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
