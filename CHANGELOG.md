# Changelog

## 0.1.3 - 2026-03-26

### Resumen
- Se añadió un resolver explícito de perfiles de runtime para Windows/NVIDIA con matriz binaria estable, evitando instalar `torch` genérico por defecto.
- El bootstrap ahora prioriza `win-cu128-stable` (con fallback `win-cu126-stable` solo si hay wheels binarias cp311/win_amd64 verificadas en PyG).
- Se bloqueó el fallback silencioso a source-build para PyG por defecto (`--only-binary=:all:`), con override experto `MODLY_UNIRIG_ALLOW_SOURCE_BUILDS=1`.
- Se añadió reparación parcial del runtime roto (desinstalación selectiva de stack torch/PyG incompatible) sin recrear Python/repo/venv completos.
- Se reforzó la validación final del bootstrap con imports explícitos de `torch`, `torch_scatter`, `torch_cluster` y `spconv`, persistiendo el resultado en estado.
- Se amplió `bootstrap_state.json` con metadata del perfil seleccionado, URLs de índices/wheels, modo binario y política de source builds.

### Archivos cambiados
- `generator.py`
- `tests/test_runtime_lifecycle.py`
- `manifest.json`
- `CHANGELOG.md`

## 0.1.2 - 2026-03-25

### Resumen
- El bootstrap ahora resuelve Python 3.11 con prioridad local-first: overrides explícitos, Python bundled de Modly, Python standalone local del runtime, `py -3.11` y finalmente PATH.
- Se añadió provisionado automático de Python 3.11 standalone (`python-build-standalone` 3.11.9 / release 20240726) dentro de `runtime/python311`.
- Se amplió `bootstrap_state.json` con trazabilidad completa del Python elegido, intentos de búsqueda y fases de instalación.
- Se añadió soporte offline para bootstrap mediante overrides de archivo/URL de Python standalone y wheelhouse local de pip.

### Archivos cambiados
- `generator.py`
- `tests/test_runtime_lifecycle.py`
- `manifest.json`
- `CHANGELOG.md`

### Comandos QA ejecutados
- `python -m unittest tests/test_runtime_lifecycle.py`

### Limitaciones pendientes
- La descarga real de `python-build-standalone` no se prueba end-to-end en tests; la red/extracción se valida vía diseño y tests unitarios con mocks.
