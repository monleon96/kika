# Sistema Híbrido de Visualización

## Descripción General

El ACE Viewer ahora utiliza un **sistema híbrido de visualización** que combina lo mejor de dos mundos:

1. **Plotly** para preview interactivo en tiempo real
2. **Matplotlib + PlotBuilder** para exportación de alta calidad

## 🎨 Plotly: Preview Interactivo

### Características
- ✅ **Velocidad**: Actualizaciones instantáneas al cambiar parámetros
- ✅ **Interactividad**: Zoom, pan, hover tooltips nativos
- ✅ **Responsive**: Se adapta al tamaño de la ventana
- ✅ **Estilo mejorado**: Template publication-quality con:
  - Fuentes serif (Computer Modern / Times New Roman)
  - Colores color-blind friendly (paleta científica)
  - Bordes y ejes estilo Matplotlib
  - Grid con estilo dot

### Uso
Simplemente configura tus series y ajusta los parámetros en tiempo real. Los cambios se reflejan inmediatamente en el preview.

## 📊 Matplotlib: Exportación de Alta Calidad

### Características
- ✅ **Calidad profesional**: Usa el PlotBuilder original de MCNPy
- ✅ **Configuración completa**: Todos los parámetros de `_plot_settings.py`
- ✅ **Múltiples formatos**: PNG, PDF, SVG
- ✅ **DPI personalizable**: 150-1200 DPI
- ✅ **Estilos disponibles**:
  - `publication`: Recomendado para papers (colores color-blind friendly)
  - `presentation`: Para presentaciones (líneas más gruesas)
  - `dark`: Fondo oscuro
  - `default`: Estilo estándar

### Uso
1. Configura tu plot en el preview interactivo de Plotly
2. Cuando estés satisfecho, haz clic en **"Export (High Quality)"**
3. Selecciona:
   - **Style**: publication, presentation, dark, o default
   - **Format**: PNG (raster), PDF (vector para papers), SVG (vector editable)
   - **DPI**: 300 DPI es el estándar para publicaciones
4. Haz clic en **Export**
5. El backend generará el plot con Matplotlib y lo descargará

## 🔄 Flujo de Trabajo Recomendado

```
1. Carga archivos ACE
   ↓
2. Añade series y configura parámetros
   ↓
3. Explora y ajusta en tiempo real con Plotly
   ↓
4. Cuando esté perfecto → Export (High Quality)
   ↓
5. Descarga imagen de alta calidad con Matplotlib
```

## 📐 Configuración de Estilos

### Plotly Preview
El preview usa automáticamente un template publication-quality:
- Fuentes serif profesionales
- Colores científicos (color-blind friendly)
- Ejes con bordes y ticks externos
- Grid sutil estilo Matplotlib

### Matplotlib Export

#### Publication Style (Recomendado)
```
- Colores: Color-blind friendly palette
- Fuentes: Serif (publication quality)
- DPI: 300 (journal standard)
- Formato: PDF o PNG
```

#### Presentation Style
```
- Líneas más gruesas (3.0 pt)
- Fuentes más grandes
- DPI: 150-300
- Formato: PNG
```

## 🎯 Ventajas del Sistema Híbrido

| Aspecto | Plotly (Preview) | Matplotlib (Export) |
|---------|------------------|---------------------|
| **Velocidad** | ⚡ Instantáneo | 🐢 2-5 segundos |
| **Interactividad** | ✅ Total | ❌ Estática |
| **Calidad científica** | ⭐⭐⭐ Buena | ⭐⭐⭐⭐⭐ Excelente |
| **Formatos** | PNG, SVG, JPEG | PNG, PDF, SVG |
| **Control fino** | ⭐⭐⭐ Medio | ⭐⭐⭐⭐⭐ Total |
| **Uso recomendado** | Exploración | Publicación |

## 💡 Consejos

1. **Para explorar datos**: Usa el preview de Plotly, es instantáneo
2. **Para papers**: Exporta con Matplotlib en PDF @ 300 DPI (publication style)
3. **Para presentaciones**: Exporta con Matplotlib en PNG @ 150 DPI (presentation style)
4. **Para editar**: Exporta en SVG (vector) para editar en Inkscape/Illustrator

## 🔧 Parámetros Soportados

Todos los parámetros que configuraste en el preview de Plotly se transfieren a Matplotlib:

- ✅ Títulos y etiquetas
- ✅ Escalas logarítmicas
- ✅ Límites de ejes (zoom)
- ✅ Grid
- ✅ Leyenda y posición
- ✅ Colores de series
- ✅ Estilos de línea
- ✅ Markers
- ✅ Grosor de línea

## 📦 Arquitectura Técnica

```
Frontend (React + Tauri)
  ├─ PlotViewer Component
  │  ├─ Plotly (react-plotly.js)
  │  │  └─ Real-time interactive preview
  │  └─ Export Button
  │     └─ Trigger Matplotlib export
  │
Backend (FastAPI + Python)
  └─ /api/plot/matplotlib-export
     ├─ Recibe configuración completa
     ├─ Usa PlotBuilder + PlotData
     ├─ Aplica estilos de _plot_settings.py
     └─ Devuelve imagen base64

```

## 🚀 Mejoras Futuras Potenciales

- [ ] Batch export (múltiples plots a la vez)
- [ ] Templates personalizados guardados
- [ ] Comparación lado a lado (Plotly vs Matplotlib)
- [ ] Export directo a LaTeX con código incluido
