# Guía de Prueba del Sistema Híbrido

## 🚀 Inicio Rápido

### 1. Iniciar el Backend (MCNPy Server)

```bash
cd /home/MONLEON-JUAN/MCNPy/mcnpy_server
python app.py
```

El servidor debería iniciar en `http://localhost:8001`

### 2. Iniciar el Frontend (Tauri Desktop)

En otra terminal:

```bash
cd /home/MONLEON-JUAN/MCNPy/kika-desktop
npm run tauri dev
```

## ✅ Pasos de Prueba

### Test 1: Verificar Estilos Mejorados de Plotly

1. Abre la app de escritorio
2. Ve a la pestaña **"ACE Viewer"**
3. Carga un archivo ACE
4. Añade una serie (botón "Add Series")
5. **Observa el preview**:
   - ✅ ¿Usa fuentes serif?
   - ✅ ¿Los ejes tienen bordes negros?
   - ✅ ¿Los ticks están afuera de los ejes?
   - ✅ ¿El grid es sutil y con puntos?
   - ✅ ¿Los colores son diferentes a los antiguos de Plotly?

### Test 2: Interactividad en Tiempo Real

1. Con una serie cargada:
   - Cambia el color → Debería actualizarse instantáneamente
   - Cambia el grosor de línea → Actualización inmediata
   - Activa markers → Se muestran al instante
   - Cambia título/labels → Actualización en tiempo real
   - Activa/desactiva log scales → Cambio inmediato
   - Ajusta límites de ejes → Zoom instantáneo

2. **Verifica que NO hay lag** en ningún cambio

### Test 3: Exportación con Matplotlib (PNG)

1. Con tu plot configurado:
2. Haz clic en **"Export (High Quality)"** (botón morado)
3. Debería abrirse un diálogo
4. Configura:
   - Style: **Publication**
   - Format: **PNG**
   - DPI: **300**
5. Haz clic en **"Export"**
6. Espera 2-5 segundos
7. **Verifica**:
   - ✅ Se descarga un archivo `ace_plot_hq.png`
   - ✅ Aparece notificación de éxito
   - ✅ La notificación muestra las dimensiones (ej: "3000×1740px @ 300 DPI")
8. Abre la imagen descargada:
   - ✅ ¿Es de alta calidad?
   - ✅ ¿Coincide con el preview?
   - ✅ ¿Usa fuentes serif?
   - ✅ ¿Se ven los markers si los activaste?

### Test 4: Exportación PDF (Vector)

1. Repite Test 3 pero con:
   - Format: **PDF**
   - DPI: **300**
2. **Verifica**:
   - ✅ Se descarga `ace_plot_hq.pdf`
   - ✅ Al hacer zoom en el PDF, las líneas NO se pixelan
   - ✅ Texto nítido a cualquier zoom

### Test 5: Estilo "Presentation"

1. Exporta con:
   - Style: **Presentation**
   - Format: **PNG**
   - DPI: **150**
2. **Verifica**:
   - ✅ Las líneas son más gruesas que en "Publication"
   - ✅ Las fuentes son más grandes
   - ✅ Mejor legibilidad para presentaciones

### Test 6: Múltiples Series

1. Añade 3-4 series diferentes
2. Configura cada una con:
   - Color distinto
   - Algunos con markers, otros sin markers
   - Diferentes estilos de línea
3. **Preview en Plotly**:
   - ✅ Todos se muestran correctamente
   - ✅ La leyenda muestra todas las series
4. **Exporta con Matplotlib**:
   - ✅ Todas las series aparecen
   - ✅ Los colores coinciden
   - ✅ Los markers coinciden
   - ✅ Los estilos de línea coinciden

### Test 7: Límites de Ejes (Zoom)

1. Configura X Min/Max y Y Min/Max
2. **Preview en Plotly**:
   - ✅ El zoom se aplica correctamente
3. **Exporta con Matplotlib**:
   - ✅ Los mismos límites se aplican
   - ✅ La imagen exportada coincide con el preview

### Test 8: Escalas Logarítmicas

1. Activa "Log X" y "Log Y"
2. **Preview en Plotly**:
   - ✅ Ambos ejes usan escala logarítmica
3. **Exporta con Matplotlib**:
   - ✅ Los ejes mantienen escala logarítmica
   - ✅ Los ticks están bien espaciados

### Test 9: Grid y Leyenda

1. Desactiva el grid
2. Cambia posición de leyenda a "bottom-left"
3. **Preview en Plotly**:
   - ✅ No se ve grid
   - ✅ Leyenda en esquina inferior izquierda
4. **Exporta con Matplotlib**:
   - ✅ Sin grid
   - ✅ Leyenda en la misma posición

### Test 10: Labels Personalizadas

1. Añade una serie
2. Cambia label mode a "Custom"
3. Escribe un label personalizado: "My Custom Label"
4. **Preview en Plotly**:
   - ✅ La leyenda muestra el label personalizado
5. **Exporta con Matplotlib**:
   - ✅ El label personalizado aparece en la exportación

## 🐛 Problemas Comunes

### El servidor no inicia
```bash
# Verifica que el puerto 8001 esté libre
lsof -i :8001

# Si está ocupado, mata el proceso o cambia el puerto en app.py
```

### Error "Failed to export with Matplotlib"
- Verifica que el servidor esté corriendo
- Revisa la consola del servidor para errores detallados
- Verifica que todos los archivos ACE estén cargados

### Los colores no coinciden exactamente
- Esto es normal: Plotly y Matplotlib renderizan colores ligeramente diferentes
- La diferencia debería ser mínima

### La exportación es muy lenta
- Exportaciones @ 600-1200 DPI tardan más (5-10 segundos)
- Considera usar 300 DPI para pruebas

## ✨ Características a Validar

### Estilos de Plotly Mejorados
- [ ] Fuentes serif profesionales
- [ ] Colores científicos color-blind friendly
- [ ] Ejes con bordes negros
- [ ] Ticks externos
- [ ] Grid sutil

### Exportación Matplotlib
- [ ] PNG de alta calidad
- [ ] PDF vectorial
- [ ] SVG vectorial
- [ ] Múltiples estilos (publication, presentation, dark)
- [ ] DPI configurable
- [ ] Transferencia correcta de todos los parámetros

### Funcionalidad
- [ ] Actualización en tiempo real sin lag
- [ ] Múltiples series simultáneas
- [ ] Markers configurables
- [ ] Estilos de línea
- [ ] Zoom/límites de ejes
- [ ] Escalas logarítmicas
- [ ] Labels personalizadas
- [ ] Grid on/off
- [ ] Leyenda configurable

## 📊 Resultado Esperado

Al final de las pruebas deberías:

1. ✅ Tener un preview rápido e interactivo en Plotly con estilo profesional
2. ✅ Poder exportar plots de alta calidad con Matplotlib
3. ✅ Ver consistencia entre preview y exportación
4. ✅ Tener flexibilidad para diferentes formatos y estilos

## 📝 Reporte de Issues

Si encuentras problemas:

1. Describe el test que falló
2. Captura de pantalla del error
3. Log de la consola del navegador (F12)
4. Log de la consola del servidor
5. Pasos para reproducir

## 🎉 ¡Listo para Producción!

Si todos los tests pasan, el sistema híbrido está listo para usar en tu flujo de trabajo diario.
