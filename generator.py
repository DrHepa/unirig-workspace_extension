from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from services.workspace_tools_base import BaseWorkspaceTool, WorkspaceToolError

TOOL_ID = 'unirig-workspace-v1'
BOOTSTRAP_VERSION = 12

TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu128'
PYG_INDEX_URL = 'https://data.pyg.org/whl/torch-2.7.0+cu128.html'
FLASH_ATTN_WHEEL_DEFAULT = (
    'https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/'
    'flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl'
)
TRITON_WINDOWS_PACKAGE_DEFAULT = 'triton-windows==3.3.1.post19'

TORCH_PACKAGES = ['torch==2.7.0', 'torchvision==0.22.0', 'torchaudio==2.7.0']
SPCONV_PACKAGE = 'spconv-cu120'
PYG_PACKAGES = ['torch_scatter==2.1.2+pt27cu128', 'torch_cluster==1.6.3+pt27cu128']
NUMPY_PIN = 'numpy==1.26.4'

DIRECT_INPUT_SUFFIXES = {'.obj', '.fbx', '.glb', '.vrm'}
CONVERTIBLE_INPUT_SUFFIXES = {'.gltf', '.stl', '.ply'}
SUPPORTED_SUFFIXES = DIRECT_INPUT_SUFFIXES | CONVERTIBLE_INPUT_SUFFIXES

SKELETON_TASK = 'configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml'
SKIN_TASK = 'configs/task/quick_inference_unirig_skin.yaml'
SKIN_DATA_NAME = 'raw_data.npz'
MERGE_REQUIRE_SUFFIX = 'obj,fbx,FBX,dae,glb,gltf,vrm'

REQUIRED_IMPORTS: list[str] = [
    'lightning',
    'pytorch_lightning',
    'transformers',
    'box',
    'einops',
    'omegaconf',
    'timm',
    'trimesh',
    'open3d',
    'pyrender',
    'huggingface_hub',
    'wandb',
    'bpy',
    'torch',
    'torch_scatter',
    'torch_cluster',
    'spconv',
    'flash_attn',
    'triton',
]

DEEP_IMPORT_CHECKS: dict[str, str] = {
    'flash_attn.layers.rotary': 'from flash_attn.layers.rotary import apply_rotary_emb',
    'transformers.models.opt.modeling_opt': 'import importlib; importlib.import_module("transformers.models.opt.modeling_opt")',
}

RUNTIME_ENV_DEFAULTS: dict[str, str] = {
    'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
    'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
}


@dataclass
class RuntimeContext:
    runtime_root: Path
    runtime_vendor_dir: Path
    venv_dir: Path
    python_exe: Path
    logs_dir: Path
    extension_root: Path
    extension_vendor_dir: Path
    active_vendor_dir: Path
    unirig_dir: Path


def _report(progress_cb: Optional[Callable[[int, str], None]], pct: int, message: str) -> None:
    if progress_cb:
        progress_cb(int(pct), message)


class UniRigWorkspaceTool(BaseWorkspaceTool):
    TOOL_ID = TOOL_ID
    DISPLAY_NAME = 'UniRig Workspace Tool'
    TOOL_KIND = 'mesh_rigger'
    PRIORITY = 100

    def __init__(self, workspace_dir: Path) -> None:
        super().__init__(workspace_dir)
        self._runtime: RuntimeContext | None = None

    def runtime_status(self) -> dict:
        runtime = _resolve_runtime_context()
        state = _load_state(runtime)
        runtime.active_vendor_dir, runtime.unirig_dir = _resolve_active_vendor_dirs(runtime)
        if state.get('install_state') == 'ready' and not _runtime_ready(runtime):
            state['install_state'] = 'error'
            state['last_error'] = 'runtime files missing or vendor incomplete'
            _save_state(runtime, state)
        return state

    def install_runtime(self, progress_cb: Optional[Callable[[int, str], None]] = None) -> dict:
        runtime = _resolve_runtime_context()
        runtime.runtime_root.mkdir(parents=True, exist_ok=True)
        runtime.logs_dir.mkdir(parents=True, exist_ok=True)
        state = _default_state(runtime)
        state.update({'install_state': 'installing', 'step': 'init', 'percent': 0})
        _save_state(runtime, state)

        try:
            _ensure_windows_binary_path()

            _report(progress_cb, 5, 'creating venv')
            _create_venv(runtime)
            _update_state(runtime, state, 15, 'venv ready')

            _report(progress_cb, 20, 'building vendor')
            _ensure_vendor(runtime)
            _update_state(runtime, state, 30, 'vendor ready')

            pip = [str(runtime.python_exe), '-m', 'pip']
            _run(pip + ['install', '--upgrade', 'pip', 'setuptools', 'wheel'])

            _report(progress_cb, 40, 'installing torch cu128')
            _run(pip + ['install', '--index-url', TORCH_INDEX_URL, *TORCH_PACKAGES], cwd=runtime.unirig_dir)
            _update_state(runtime, state, 40, 'torch installed')

            _report(progress_cb, 48, 'installing triton runtime')
            _install_triton(runtime)
            _update_state(runtime, state, 48, 'triton installed')

            _report(progress_cb, 55, 'installing UniRig requirements')
            _install_official_requirements_excluding_flash_attn(runtime)
            _update_state(runtime, state, 55, 'requirements installed')

            _report(progress_cb, 68, 'installing spconv')
            _run(pip + ['install', SPCONV_PACKAGE], cwd=runtime.unirig_dir)
            _update_state(runtime, state, 68, 'spconv installed')

            _report(progress_cb, 76, 'installing torch_scatter/torch_cluster')
            _run(pip + ['install', '-f', PYG_INDEX_URL, '--no-cache-dir', *PYG_PACKAGES], cwd=runtime.unirig_dir)
            _update_state(runtime, state, 76, 'pyg deps installed')

            _report(progress_cb, 82, 'pinning numpy')
            _run(pip + ['install', NUMPY_PIN], cwd=runtime.unirig_dir)
            _update_state(runtime, state, 82, 'numpy pinned')

            _report(progress_cb, 88, 'installing flash-attn wheel')
            _install_flash_attn(runtime)
            _update_state(runtime, state, 88, 'flash-attn installed')

            py_path = _vendor_pythonpath(runtime)
            _report(progress_cb, 94, 'validating imports')
            missing = _missing_imports(runtime, py_path)
            if missing:
                raise WorkspaceToolError('missing imports: ' + '; '.join(missing))

            _report(progress_cb, 97, 'validating run.py --help')
            _run([str(runtime.python_exe), 'run.py', '--help'], cwd=runtime.unirig_dir, extra_env={'PYTHONPATH': py_path})

            state.update({'install_state': 'ready', 'step': 'ready', 'percent': 100, 'last_error': '', 'updated_at': int(time.time())})
            _save_state(runtime, state)
            self._runtime = runtime
            _report(progress_cb, 100, 'ready')
            return state
        except Exception as exc:
            state.update({'install_state': 'error', 'step': 'failed', 'last_error': str(exc), 'updated_at': int(time.time())})
            _save_state(runtime, state)
            raise WorkspaceToolError(f'Failed to install UniRig runtime: {exc}') from exc

    def uninstall_runtime(self) -> None:
        runtime = _resolve_runtime_context()
        if runtime.runtime_root.exists():
            shutil.rmtree(runtime.runtime_root, ignore_errors=True)

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        runtime = self._runtime or _resolve_runtime_context()
        _ensure_vendor(runtime)
        status = self.runtime_status()
        if status.get('install_state') != 'ready' or not _runtime_ready(runtime):
            raise WorkspaceToolError('UniRig runtime is not ready. Install runtime first.')

        py_path = _vendor_pythonpath(runtime)
        missing = _ensure_runtime_dependencies(runtime, py_path)
        if missing:
            raise WorkspaceToolError('UniRig runtime is missing dependencies. Run Repair runtime. Missing: ' + '; '.join(missing))

        suffix = input_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise WorkspaceToolError(f'Unsupported mesh format: {suffix}')

        seed = int(params.get('seed', 12345))
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        output_path = output_dir / f'{input_path.stem}_unirig_{ts}.glb'

        with tempfile.TemporaryDirectory(prefix='unirig_stage_') as tmp:
            stage_root = Path(tmp)
            prepared = _prepare_input_mesh(input_path, stage_root)
            skeleton_output = stage_root / 'skeleton_stage.fbx'
            skin_output = stage_root / 'skin_stage.fbx'
            skeleton_npz_dir = stage_root / 'skeleton_npz'
            skin_npz_dir = stage_root / 'skin_npz'

            _report(progress_cb, 50, 'predicting skeleton')
            _run(
                [
                    str(runtime.python_exe),
                    'run.py',
                    f'--task={SKELETON_TASK}',
                    f'--seed={seed}',
                    f'--input={prepared}',
                    f'--output={skeleton_output}',
                    f'--npz_dir={skeleton_npz_dir}',
                ],
                cwd=runtime.unirig_dir,
                extra_env={'PYTHONPATH': py_path},
            )
            if not skeleton_output.exists():
                raise WorkspaceToolError('Skeleton stage did not produce output.')

            _report(progress_cb, 72, 'predicting skin')
            _run(
                [
                    str(runtime.python_exe),
                    'run.py',
                    f'--task={SKIN_TASK}',
                    f'--seed={seed}',
                    f'--input={skeleton_output}',
                    f'--output={skin_output}',
                    f'--npz_dir={skin_npz_dir}',
                    f'--data_name={SKIN_DATA_NAME}',
                ],
                cwd=runtime.unirig_dir,
                extra_env={'PYTHONPATH': py_path},
            )
            if not skin_output.exists():
                raise WorkspaceToolError('Skin stage did not produce output.')

            _report(progress_cb, 90, 'merging rig')
            _run(
                [
                    str(runtime.python_exe),
                    '-m',
                    'src.inference.merge',
                    f'--require_suffix={MERGE_REQUIRE_SUFFIX}',
                    '--num_runs=1',
                    '--id=0',
                    f'--source={skin_output}',
                    f'--target={prepared}',
                    f'--output={output_path}',
                ],
                cwd=runtime.unirig_dir,
                extra_env={'PYTHONPATH': py_path},
            )

        if not output_path.exists():
            raise WorkspaceToolError('Merge stage did not produce output GLB.')
        _report(progress_cb, 100, 'done')
        return output_path


def _state_path(runtime: RuntimeContext) -> Path:
    return runtime.runtime_root / 'bootstrap_state.json'


def _default_state(runtime: RuntimeContext) -> dict:
    now = int(time.time())
    return {
        'bootstrap_version': BOOTSTRAP_VERSION,
        'install_state': 'not_installed',
        'step': 'idle',
        'percent': 0,
        'last_error': '',
        'updated_at': now,
        'runtime_root': str(runtime.runtime_root),
        'vendor_dir': str(runtime.active_vendor_dir),
        'unirig_dir': str(runtime.unirig_dir),
        'venv_dir': str(runtime.venv_dir),
        'python_exe': str(runtime.python_exe),
    }


def _load_state(runtime: RuntimeContext) -> dict:
    path = _state_path(runtime)
    if not path.exists():
        return _default_state(runtime)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return _default_state(runtime)


def _save_state(runtime: RuntimeContext, state: dict) -> None:
    runtime.active_vendor_dir, runtime.unirig_dir = _resolve_active_vendor_dirs(runtime)
    state.update(
        {
            'bootstrap_version': BOOTSTRAP_VERSION,
            'runtime_root': str(runtime.runtime_root),
            'vendor_dir': str(runtime.active_vendor_dir),
            'unirig_dir': str(runtime.unirig_dir),
            'venv_dir': str(runtime.venv_dir),
            'python_exe': str(runtime.python_exe),
        }
    )
    path = _state_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding='utf-8')


def _update_state(runtime: RuntimeContext, state: dict, percent: int, step: str) -> None:
    state['percent'] = percent
    state['step'] = step
    state['updated_at'] = int(time.time())
    _save_state(runtime, state)


def _resolve_runtime_context() -> RuntimeContext:
    extension_root = Path(__file__).resolve().parent
    extension_vendor_dir = extension_root / 'vendor'

    override = os.environ.get('MODLY_UNIRIG_RUNTIME_DIR', '').strip()
    if override:
        runtime_root = Path(override).expanduser().resolve()
    else:
        user_data = os.environ.get('MODLY_USERDATA_DIR', '').strip()
        if user_data:
            runtime_root = Path(user_data).expanduser().resolve() / 'dependencies' / TOOL_ID
        else:
            runtime_root = Path.home() / '.cache' / 'modly' / TOOL_ID

    runtime_vendor_dir = runtime_root / 'vendor'
    venv_dir = runtime_root / 'venv'
    python_exe = venv_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    logs_dir = runtime_root / 'logs'

    runtime = RuntimeContext(
        runtime_root=runtime_root,
        runtime_vendor_dir=runtime_vendor_dir,
        venv_dir=venv_dir,
        python_exe=python_exe,
        logs_dir=logs_dir,
        extension_root=extension_root,
        extension_vendor_dir=extension_vendor_dir,
        active_vendor_dir=runtime_vendor_dir,
        unirig_dir=runtime_vendor_dir / 'unirig',
    )
    runtime.active_vendor_dir, runtime.unirig_dir = _resolve_active_vendor_dirs(runtime)
    return runtime


def _resolve_active_vendor_dirs(runtime: RuntimeContext) -> tuple[Path, Path]:
    if _validate_vendor_dir(runtime.runtime_vendor_dir):
        return runtime.runtime_vendor_dir, runtime.runtime_vendor_dir / 'unirig'
    if _validate_vendor_dir(runtime.extension_vendor_dir):
        return runtime.extension_vendor_dir, runtime.extension_vendor_dir / 'unirig'
    return runtime.runtime_vendor_dir, runtime.runtime_vendor_dir / 'unirig'


def _required_vendor_paths(vendor_dir: Path) -> list[Path]:
    return [
        vendor_dir / 'unirig' / 'run.py',
        vendor_dir / 'unirig' / 'src',
        vendor_dir / 'unirig' / 'configs',
        vendor_dir / 'unirig' / 'requirements.txt',
    ]


def _validate_vendor_dir(vendor_dir: Path) -> bool:
    return all(path.exists() for path in _required_vendor_paths(vendor_dir))


def _ensure_vendor(runtime: RuntimeContext) -> None:
    if not _validate_vendor_dir(runtime.runtime_vendor_dir) and not _validate_vendor_dir(runtime.extension_vendor_dir):
        _build_runtime_vendor(runtime)

    runtime.active_vendor_dir, runtime.unirig_dir = _resolve_active_vendor_dirs(runtime)
    if not _validate_vendor_dir(runtime.active_vendor_dir):
        required = ', '.join(str(p) for p in _required_vendor_paths(runtime.active_vendor_dir))
        raise WorkspaceToolError(f'vendor/ is incomplete. Missing: {required}')


def _build_runtime_vendor(runtime: RuntimeContext) -> None:
    runtime.runtime_vendor_dir.parent.mkdir(parents=True, exist_ok=True)
    build_vendor_py = runtime.extension_root / 'build_vendor.py'
    if not build_vendor_py.exists():
        raise WorkspaceToolError('build_vendor.py is missing from the extension root.')
    cmd = [sys.executable, str(build_vendor_py), '--dest', str(runtime.runtime_vendor_dir)]
    source_zip = os.environ.get('MODLY_UNIRIG_SOURCE_ZIP', '').strip()
    if source_zip:
        cmd.extend(['--source-zip', source_zip])
    ref = os.environ.get('MODLY_UNIRIG_REPO_REF', '').strip()
    if ref:
        cmd.extend(['--ref', ref])
    _run(cmd, cwd=runtime.extension_root)


def _vendor_pythonpath(runtime: RuntimeContext) -> str:
    current = os.environ.get('PYTHONPATH', '').strip()
    parts = [str(runtime.active_vendor_dir), str(runtime.unirig_dir)]
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def _ensure_windows_binary_path() -> None:
    if os.name != 'nt':
        raise WorkspaceToolError('This runtime profile supports Windows + NVIDIA only.')


def _runtime_ready(runtime: RuntimeContext) -> bool:
    runtime.active_vendor_dir, runtime.unirig_dir = _resolve_active_vendor_dirs(runtime)
    state = _load_state(runtime)
    return state.get('install_state') == 'ready' and runtime.python_exe.exists() and _validate_vendor_dir(runtime.active_vendor_dir)


def _python_cmd_is_311(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(cmd + ['-c', 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'], capture_output=True, text=True, timeout=20)
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip() == '3.11'


def _resolve_python311_cmd() -> list[str]:
    env_bin = os.environ.get('MODLY_UNIRIG_PYTHON311_BIN', '').strip()
    candidates: list[list[str]] = []
    if env_bin:
        candidates.append([env_bin])

    if sys.version_info[:2] == (3, 11) and sys.executable:
        candidates.append([sys.executable])

    modly_python = os.environ.get('MODLY_PYTHON_EXE', '').strip()
    if modly_python:
        candidates.append([modly_python])

    for raw in [shutil.which('python3.11'), shutil.which('python')]:
        if raw:
            candidates.append([raw])
    if shutil.which('py'):
        candidates.append(['py', '-3.11'])

    seen: set[tuple[str, ...]] = set()
    for cmd in candidates:
        key = tuple(cmd)
        if key in seen:
            continue
        seen.add(key)
        if _python_cmd_is_311(cmd):
            return cmd
    raise WorkspaceToolError('Python 3.11 runtime not found. Set MODLY_UNIRIG_PYTHON311_BIN or run Modly with its bundled Python 3.11 backend.')


def _create_venv(runtime: RuntimeContext) -> None:
    if runtime.python_exe.exists():
        return
    runtime.venv_dir.parent.mkdir(parents=True, exist_ok=True)
    py311_cmd = _resolve_python311_cmd()
    _run([*py311_cmd, '-m', 'venv', str(runtime.venv_dir)], cwd=runtime.runtime_root)


def _triton_package() -> str:
    raw = os.environ.get('MODLY_UNIRIG_TRITON_PACKAGE', TRITON_WINDOWS_PACKAGE_DEFAULT).strip()
    return raw or TRITON_WINDOWS_PACKAGE_DEFAULT


def _install_triton(runtime: RuntimeContext) -> None:
    if os.name != 'nt':
        return
    _run([str(runtime.python_exe), '-m', 'pip', 'install', _triton_package()], cwd=runtime.unirig_dir)


def _install_official_requirements_excluding_flash_attn(runtime: RuntimeContext) -> None:
    req_path = runtime.unirig_dir / 'requirements.txt'
    if not req_path.exists():
        raise WorkspaceToolError(f'Missing vendored requirements.txt at {req_path}')
    filtered_path = runtime.runtime_root / 'requirements.unirig.filtered.txt'
    lines = []
    for raw in req_path.read_text(encoding='utf-8').splitlines():
        if 'flash_attn' in raw.replace('-', '_').lower():
            continue
        lines.append(raw)
    filtered_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    _run([str(runtime.python_exe), '-m', 'pip', 'install', '-r', str(filtered_path)], cwd=runtime.unirig_dir)


def _install_flash_attn(runtime: RuntimeContext) -> None:
    wheel_url = os.environ.get('MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL', FLASH_ATTN_WHEEL_DEFAULT).strip() or FLASH_ATTN_WHEEL_DEFAULT
    cache_dir = runtime.runtime_root / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = wheel_url.rsplit('/', 1)[-1].split('?', 1)[0]
    wheel_path = cache_dir / wheel_name
    try:
        if not wheel_path.exists():
            urllib.request.urlretrieve(wheel_url, str(wheel_path))
        _run([str(runtime.python_exe), '-m', 'pip', 'install', str(wheel_path)])
    except Exception as exc:
        _append_log(runtime, 'flash_attn_install.log', f'wheel_url={wheel_url}\nerror={exc}')
        raise WorkspaceToolError('flash-attn wheel install failed') from exc


def _append_log(runtime: RuntimeContext, filename: str, detail: str) -> None:
    runtime.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = runtime.logs_dir / filename
    with log_file.open('a', encoding='utf-8') as handle:
        handle.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}] {detail}\n')


def _missing_imports(runtime: RuntimeContext, py_path: str) -> list[str]:
    checks: list[tuple[str, str]] = []
    for module_name in REQUIRED_IMPORTS:
        checks.append((module_name, f'import importlib; importlib.import_module("{module_name}")'))
    checks.extend(DEEP_IMPORT_CHECKS.items())

    code = (
        'import json\n'
        f'checks={checks!r}\n'
        'failed=[]\n'
        'for label, stmt in checks:\n'
        '  try:\n'
        '    exec(stmt, {})\n'
        '  except Exception as exc:\n'
        '    failed.append({"check": label, "error": str(exc)})\n'
        'print(json.dumps({"failed": failed}))\n'
    )
    output = _run_capture([str(runtime.python_exe), '-c', code], extra_env={'PYTHONPATH': py_path})
    payload = json.loads(output.strip().splitlines()[-1])
    return [f"{item['check']}: {item['error']}" for item in payload.get('failed', [])]


def _ensure_runtime_dependencies(runtime: RuntimeContext, py_path: str) -> list[str]:
    missing = _missing_imports(runtime, py_path)
    if not missing:
        return []
    recoverable = any(
        entry.startswith('triton:')
        or entry.startswith('flash_attn.layers.rotary:')
        or entry.startswith('transformers.models.opt.modeling_opt:')
        for entry in missing
    )
    if os.name == 'nt' and recoverable:
        _install_triton(runtime)
        missing = _missing_imports(runtime, py_path)
    return missing


def _prepare_input_mesh(input_path: Path, temp_root: Path) -> Path:
    suffix = input_path.suffix.lower()
    if suffix in DIRECT_INPUT_SUFFIXES:
        return input_path.resolve()
    if suffix not in CONVERTIBLE_INPUT_SUFFIXES:
        raise WorkspaceToolError(f'Unsupported input format: {suffix}')

    import trimesh

    converted_path = temp_root / f'{input_path.stem}_prepared.glb'
    loaded = trimesh.load(input_path, force='scene' if suffix == '.gltf' else None)
    scene = loaded.scene() if isinstance(loaded, trimesh.Trimesh) else loaded
    blob = scene.export(file_type='glb')
    converted_path.write_bytes(blob if isinstance(blob, (bytes, bytearray)) else bytes(blob))
    return converted_path


def _compose_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in RUNTIME_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    if extra_env:
        env.update(extra_env)
    return env


def _run(command: list[str], cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> None:
    env = _compose_env(extra_env)
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '').strip().splitlines()[-60:]
        raise WorkspaceToolError(f'Command failed: {" ".join(command)}\n' + '\n'.join(tail))


def _run_capture(command: list[str], cwd: Path | None = None, extra_env: dict[str, str] | None = None) -> str:
    env = _compose_env(extra_env)
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '').strip().splitlines()[-60:]
        raise WorkspaceToolError(f'Command failed: {" ".join(command)}\n' + '\n'.join(tail))
    return result.stdout or result.stderr or ''


__all__ = ['UniRigWorkspaceTool']
