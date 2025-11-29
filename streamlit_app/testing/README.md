# Testing Scripts

Scripts de prueba para validar la conexión con el backend de KIKA.

## Scripts disponibles

### `test_connection.py`

Prueba la conexión con el backend de KIKA.

**Uso:**
```bash
poetry run python testing/test_connection.py
```

**Qué hace:**
- Carga las variables de entorno desde `.env`
- Verifica la URL del backend configurada
- Hace health check al endpoint `/healthz`
- Confirma que el backend está operativo

**Salida esperada:**
```
==================================================
🔍 Testing Backend Connection
==================================================

Backend URL: https://kika-backend.onrender.com

Testing health check...
✅ Backend is healthy and reachable!

==================================================
✅ All tests passed!
==================================================
```

---

## Notas

- Asegúrate de tener el archivo `.env` configurado con `KIKA_BACKEND_URL`
- Los scripts requieren que las dependencias estén instaladas (`poetry install`)
