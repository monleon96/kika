# Resumen de Implementación: Sistema Híbrido de Visualización

## ✅ Implementado

### Backend (mcnpy_server/app.py)

1. **Nuevo endpoint `/api/plot/matplotlib-export`**
   - Genera plots usando Matplotlib + PlotBuilder
   - Soporta estilos: `publication`, `presentation`, `dark`, `default`
   - Formatos: PNG, PDF, SVG
   - DPI personalizable: 150-1200
   - Aplica toda la configuración de series y figure settings

2. **Funciones auxiliares**
   - `_convert_plotly_linestyle()`: Convierte estilos de línea Plotly → Matplotlib
   - `_convert_plotly_marker()`: Convierte símbolos de marker Plotly → Matplotlib

3. **Modelos Pydantic**
   - `MatplotlibExportRequest`: Configuración completa para export
   - `MatplotlibExportResponse`: Respuesta con imagen base64 y metadatos

### Frontend (kika-desktop/src)

1. **PlotViewer.tsx**
   - **Banner informativo** explicando el sistema híbrido
   - **Estilos mejorados para Plotly**:
     - Fuentes serif (Computer Modern / Times New Roman)
     - Paleta color-blind friendly (colores científicos)
     - Ejes con bordes y ticks externos estilo Matplotlib
     - Grid mejorado
   - **Nuevo botón "Export (High Quality)"**
   - **Modal de configuración de export** con:
     - Selector de estilo (publication/presentation/dark/default)
     - Selector de formato (PNG/PDF/SVG)
     - Selector de DPI (150/300/600/1200)
     - Preview de configuración
   - **Función `handleMatplotlibExport()`** que:
     - Envía toda la configuración al backend
     - Recibe la imagen en base64
     - Descarga automáticamente el archivo
     - Muestra notificación de éxito con dimensiones

2. **mcnpyService.ts**
   - Nueva interfaz `MatplotlibExportRequest`
   - Nueva interfaz `MatplotlibExportResponse`
   - Nueva función `exportWithMatplotlib()`

3. **Paleta de colores actualizada**
   - Cambiada a la misma paleta color-blind friendly que usa PlotBuilder
   - Consistencia entre Plotly preview y Matplotlib export

### Documentación

1. **PLOTTING_SYSTEM.md**
   - Guía completa del sistema híbrido
   - Ventajas y desventajas de cada método
   - Flujo de trabajo recomendado
   - Tabla comparativa
   - Consejos de uso
   - Arquitectura técnica

## 🎯 Características Principales

### Plotly (Preview Interactivo)
- ✅ Actualización en tiempo real
- ✅ Interactividad completa (zoom, pan, hover)
- ✅ Estilo publication-quality mejorado
- ✅ Colores científicos
- ✅ Fuentes serif profesionales

### Matplotlib (Exportación)
- ✅ Usa PlotBuilder original de MCNPy
- ✅ Todos los parámetros de `_plot_settings.py`
- ✅ 4 estilos predefinidos
- ✅ 3 formatos (PNG, PDF, SVG)
- ✅ DPI personalizable
- ✅ Calidad profesional para publicaciones

## 📋 Parámetros Transferidos

Todos estos parámetros configurados en Plotly se transfieren a Matplotlib:

- ✅ Título y etiquetas de ejes
- ✅ Escalas logarítmicas (X e Y)
- ✅ Límites de ejes (zoom)
- ✅ Mostrar/ocultar grid
- ✅ Mostrar/ocultar leyenda
- ✅ Posición de leyenda
- ✅ Dimensiones de la figura
- ✅ **Por cada serie**:
  - Color
  - Grosor de línea
  - Estilo de línea (solid/dash/dot/dashdot)
  - Markers (on/off + símbolo + tamaño)
  - Label personalizada

## 🔄 Flujo de Usuario

```
1. Usuario añade series
   ↓
2. Ajusta parámetros en tiempo real (Plotly)
   ↓
3. Ve preview instantáneo con estilo mejorado
   ↓
4. Hace clic en "Export (High Quality)"
   ↓
5. Selecciona estilo/formato/DPI
   ↓
6. Backend genera con Matplotlib + PlotBuilder
   ↓
7. Descarga automática de imagen HQ
```

## 📊 Comparativa

| Característica | Plotly Preview | Matplotlib Export |
|----------------|----------------|-------------------|
| Velocidad | ⚡ Instantáneo | 🐢 2-5 seg |
| Interactividad | ✅ Completa | ❌ Imagen estática |
| Calidad científica | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Control fino | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Formatos | PNG, SVG, JPEG, WebP | PNG, PDF, SVG |
| DPI máximo | ~300 | 1200 |
| Uso | Exploración | Publicación |

## 🚀 Ventajas del Sistema Híbrido

1. **No sacrificas velocidad**: Preview rápido con Plotly
2. **No sacrificas calidad**: Export profesional con Matplotlib
3. **Consistencia visual**: Plotly usa estilos similares a Matplotlib
4. **Mismos parámetros**: Lo que ves es lo que obtienes (WYSIWYG)
5. **Flexibilidad**: Diferentes formatos y resoluciones para diferentes usos

## 💡 Casos de Uso

### Para exploración de datos
- Usa Plotly preview
- Cambia parámetros en tiempo real
- Experimenta con diferentes configuraciones

### Para paper científico
- Configura el plot perfecto en Plotly
- Export → Publication style
- Formato: PDF @ 300 DPI
- Resultado: Vector de alta calidad listo para journal

### Para presentación
- Configura en Plotly
- Export → Presentation style
- Formato: PNG @ 150 DPI
- Resultado: Líneas más gruesas, legible en proyector

### Para editar después
- Configura en Plotly
- Export → Publication style
- Formato: SVG
- Resultado: Vector editable en Inkscape/Illustrator

## 🎨 Mejoras Visuales en Plotly

Antes vs Ahora:

**Antes:**
- Fuentes sans-serif genéricas
- Colores básicos de Plotly
- Ejes sin bordes
- Grid muy marcado

**Ahora:**
- ✅ Fuentes serif profesionales (Computer Modern)
- ✅ Colores color-blind friendly científicos
- ✅ Ejes con bordes negros y mirror
- ✅ Ticks externos estilo Matplotlib
- ✅ Grid sutil y profesional
- ✅ Leyenda con borde negro

## 🔧 Próximos Pasos (Opcional)

- [ ] Batch export (exportar múltiples plots)
- [ ] Templates de estilo guardables
- [ ] Vista lado a lado (Plotly vs Matplotlib)
- [ ] Export con código LaTeX incluido
- [ ] Presets para diferentes journals (Nature, Science, etc.)
