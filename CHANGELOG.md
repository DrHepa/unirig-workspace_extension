# Changelog

## 0.1.5 - 2026-03-26

### Resumen
- Se separó `flash_attn` del `pip install -r requirements.txt` genérico: ahora el bootstrap instala el requirements oficial filtrado (excluyendo solo `flash_attn`) y luego ejecuta una fase dedicada obligatoria `installing flash-attn`.
- Se añadió instalador dedicado de `flash_attn` con prerequisitos (`packaging`, `psutil`, `ninja`) y modo por spec con `--no-build-isolation`.
- El perfil `win-cu128-stable` ahora fija explícitamente `flash_attn==2.7.4.post1` (override experto: `MODLY_UNIRIG_FLASH_ATTN_SPEC`).
- Se añadieron overrides expertos para wheel local/remota (`MODLY_UNIRIG_FLASH_ATTN_WHEEL`, `MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL`) con orden de resolución local > URL > spec.
- Se incorporó preflight de toolchain en Windows antes de source install (MSVC `cl.exe`, entorno CUDA `nvcc`/`CUDA_PATH`, `ninja`) y error accionable cuando falta.
- Se mejoró el runtime resumable/repair: si solo falta `flash_attn`, reanuda en la fase dedicada sin reinstalar baseline completo.
- Se amplió `bootstrap_state.json` con estado persistido de `flash_attn` (requerido, spec, modo, versión, validación y último error).
- La validación final exige `import flash_attn` y mantiene `run.py --help`.

### Archivos cambiados
- `generator.py`
- `tests/test_runtime_lifecycle.py`
- `README.md`
- `manifest.json`
- `CHANGELOG.md`

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
