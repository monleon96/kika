# MCNPy Processing Server

Servidor FastAPI local para procesamiento de archivos ACE con MCNPy.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Iniciar el servidor

```bash
python app.py
```

El servidor estará disponible en `http://localhost:8001`

### Con uvicorn (reload automático para desarrollo)

```bash
uvicorn app:app --reload --port 8001
```

## Endpoints

### GET /healthz
Health check del servidor

### POST /api/ace/parse
Parse un archivo ACE y devuelve información básica

**Request:**
```json
{
  "file_content": "contenido del archivo ACE",
  "file_name": "nombre.ace"
}
```

**Response:**
```json
{
  "zaid": "92235.80c",
  "atomic_weight_ratio": 233.0248,
  "temperature": 293.6,
  "available_reactions": [1, 2, 4, 16, 17, 18, 102],
  "has_angular_distributions": true,
  "energy_grid_size": 35234
}
```

### POST /api/ace/plot
Genera un gráfico y lo devuelve como imagen base64

**Request:**
```json
{
  "file_content": "contenido del archivo ACE",
  "file_name": "nombre.ace",
  "plot_type": "xs",
  "mt_number": 2,
  "energy_min": 1e-5,
  "energy_max": 20.0
}
```

**Response:**
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "format": "png"
}
```

### POST /api/ace/upload
Upload directo de archivo (multipart/form-data)

## Integración con Frontend

El servidor está configurado con CORS para aceptar requests desde:
- `http://localhost:1420` (Vite dev server)
- `http://localhost:5173` (Vite alternativo)
- `tauri://localhost` (Tauri app)

## Desarrollo

El servidor usa Matplotlib con backend 'Agg' (no interactivo) para generar plots en memoria sin necesidad de display gráfico.
