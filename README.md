# UniRig Workspace Tool

Workspace-tool extension for Modly.

## Flujo normal (Windows/NVIDIA)

Desde `0.2.0`, el modo por defecto es **artifact** en Windows:

1. Resolver perfil (`win-cu128-stable`).
2. Asegurar Python 3.11 local.
3. Crear venv local.
4. Cargar runtime manifest versionado (`runtime-manifest.win-cu128-stable.json`).
5. Descargar runtime artifact desde GitHub Releases.
6. Verificar `sha256`.
7. Extraer wheelhouse local.
8. Instalar dependencias con:
   - `pip install --no-index --find-links=<wheelhouse> -r runtime-lock.txt`
9. Preparar repo UniRig fijado al ref del manifest.
10. Validar imports requeridos y `python run.py --help`.
11. Marcar `install_state=ready`.

> Esta ruta **no** compila `flash_attn` ni usa `nvcc`, `vcvars64`, `cl.exe`.

## Install mode

Hay dos modos explícitos:

- `artifact` (default en Windows)
- `source` (solo dev/advanced)

Overrides para activar source build:

- `MODLY_UNIRIG_INSTALL_MODE=source`
- `MODLY_UNIRIG_DEV_SOURCE_BOOTSTRAP=1`

Si el artifact no está disponible y no hay override source, el error es corto y claro:

- `UniRig runtime artifact unavailable`

## Repair runtime

En modo `artifact`, **Repair runtime** revalida y reinstala de forma determinista desde el wheelhouse local/artifact si falta algo. No entra a source build salvo override explícito.

## Runtime layout

- Cache runtime: `Settings.dependenciesDir/unirig-workspace-v1/` (o fallback local `_runtime/`).
- Artifact cache: `runtime_artifacts/<profile_id>/`.
- Estado persistido: `bootstrap_state.json` con campos de install mode, manifest y validación.

## Variables de entorno útiles

- `MODLY_UNIRIG_INSTALL_MODE=artifact|source`
- `MODLY_UNIRIG_DEV_SOURCE_BOOTSTRAP=1`
- `MODLY_UNIRIG_RUNTIME_MANIFEST_URL`
- `MODLY_UNIRIG_FORCE_BOOTSTRAP=1`
- `MODLY_UNIRIG_RUNTIME_DIR`
- `MODLY_UNIRIG_REPO_DIR`

Variables de source build (`MODLY_UNIRIG_FLASH_ATTN_*`, toolchain CUDA/MSVC) son solo para modo `source`.
