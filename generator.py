from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import trimesh

from services.workspace_tools_base import BaseWorkspaceTool, WorkspaceToolError

TOOL_ID = 'unirig-workspace-v1'
BOOTSTRAP_VERSION = 9
OFFICIAL_UNIRIG_REPO = 'VAST-AI-Research/UniRig'
OFFICIAL_UNIRIG_REF = os.environ.get('MODLY_UNIRIG_REPO_REF', 'main')
OFFICIAL_UNIRIG_ZIP_URL = (
    'https://github.com/' + OFFICIAL_UNIRIG_REPO + '/archive/' + OFFICIAL_UNIRIG_REF + '.zip'
)

TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu128'
PYG_INDEX_URL = 'https://data.pyg.org/whl/torch-2.7.0+cu128.html'
FLASH_ATTN_WHEEL_DEFAULT = (
    'https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/'
    'flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl'
)

TORCH_PACKAGES = [
    'torch==2.7.0',
    'torchvision==0.22.0',
    'torchaudio==2.7.0',
]
SPCONV_PACKAGE = 'spconv-cu120'
TORCH_SCATTER = 'torch_scatter==2.1.2+pt27cu128'
TORCH_CLUSTER = 'torch_cluster==1.6.3+pt27cu128'
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
]


@dataclass(frozen=True)
class RuntimeContext:
    runtime_root: Path
    repo_dir: Path
    venv_dir: Path
    python_exe: Path
    logs_dir: Path


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
        if state.get('install_state') == 'ready' and not _runtime_ready(runtime):
            state['install_state'] = 'error'
            state['last_error'] = 'runtime files missing'
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
            _report(progress_cb, 5, 'preparing runtime')
            _update_state(runtime, state, 5, 'prepare runtime')

            _prepare_repo(runtime)
            _report(progress_cb, 15, 'repo ready')
            _update_state(runtime, state, 15, 'repo ready')

            _create_venv(runtime)
            _report(progress_cb, 25, 'venv ready')
            _update_state(runtime, state, 25, 'venv ready')

            pip = [str(runtime.python_exe), '-m', 'pip']
            _run(pip + ['install', '--upgrade', 'pip', 'setuptools', 'wheel'], cwd=runtime.repo_dir)

            _report(progress_cb, 35, 'installing torch cu128')
            _run(pip + ['install', '--index-url', TORCH_INDEX_URL, *TORCH_PACKAGES], cwd=runtime.repo_dir)
            _update_state(runtime, state, 35, 'torch installed')

            _report(progress_cb, 50, 'installing unirig requirements (except flash_attn)')
            _install_requirements_excluding_flash_attn(runtime)
            _update_state(runtime, state, 50, 'requirements installed')

            _report(progress_cb, 60, 'installing spconv')
            _run(pip + ['install', SPCONV_PACKAGE], cwd=runtime.repo_dir)
            _update_state(runtime, state, 60, 'spconv installed')

            _report(progress_cb, 70, 'installing torch_scatter/torch_cluster')
            _run(pip + ['install', '-f', PYG_INDEX_URL, TORCH_SCATTER, TORCH_CLUSTER], cwd=runtime.repo_dir)
            _update_state(runtime, state, 70, 'pyg deps installed')

            _report(progress_cb, 78, 'installing numpy pin')
            _run(pip + ['install', NUMPY_PIN], cwd=runtime.repo_dir)
            _update_state(runtime, state, 78, 'numpy pinned')

            _report(progress_cb, 86, 'installing flash-attn wheel')
            _install_flash_attn(runtime)
            _update_state(runtime, state, 86, 'flash-attn installed')

            _report(progress_cb, 93, 'validating imports')
            missing = _missing_imports(runtime)
            if missing:
                raise WorkspaceToolError(f'missing imports: {", ".join(missing)}')

            _report(progress_cb, 97, 'validating run.py --help')
            _run([str(runtime.python_exe), 'run.py', '--help'], cwd=runtime.repo_dir)

            state.update(
                {
                    'install_state': 'ready',
                    'step': 'ready',
                    'percent': 100,
                    'last_error': '',
                    'updated_at': int(time.time()),
                    'python_exe': str(runtime.python_exe),
                }
            )
            _save_state(runtime, state)
            self._runtime = runtime
            _report(progress_cb, 100, 'ready')
            return state
        except Exception as exc:
            state.update(
                {
                    'install_state': 'error',
                    'step': 'failed',
                    'last_error': str(exc),
                    'updated_at': int(time.time()),
                }
            )
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
        status = self.runtime_status()
        if status.get('install_state') != 'ready' or not _runtime_ready(runtime):
            raise WorkspaceToolError('UniRig runtime is not ready. Install runtime first.')

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

            self._report(progress_cb, 50, 'Predicting skeleton…')
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
                cwd=runtime.repo_dir,
            )
            if not skeleton_output.exists():
                raise WorkspaceToolError('Skeleton stage did not produce output.')

            self._report(progress_cb, 72, 'Predicting skin…')
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
                cwd=runtime.repo_dir,
            )
            if not skin_output.exists():
                raise WorkspaceToolError('Skin stage did not produce output.')

            self._report(progress_cb, 90, 'Merging rig…')
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
                cwd=runtime.repo_dir,
            )

        if not output_path.exists():
            raise WorkspaceToolError('Merge stage did not produce output GLB.')
        self._report(progress_cb, 100, 'done')
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
        'repo_dir': str(runtime.repo_dir),
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
    path = _state_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding='utf-8')


def _update_state(runtime: RuntimeContext, state: dict, percent: int, step: str) -> None:
    state['percent'] = percent
    state['step'] = step
    state['updated_at'] = int(time.time())
    _save_state(runtime, state)


def _resolve_runtime_context() -> RuntimeContext:
    override = os.environ.get('MODLY_UNIRIG_RUNTIME_DIR')
    if override:
        runtime_root = Path(override).expanduser().resolve()
    else:
        user_data = os.environ.get('MODLY_USERDATA_DIR')
        if user_data:
            runtime_root = Path(user_data).expanduser().resolve() / 'dependencies' / TOOL_ID
        else:
            runtime_root = Path.home() / '.cache' / 'modly' / TOOL_ID
    repo_dir = runtime_root / 'repo'
    venv_dir = runtime_root / 'venv'
    python_exe = venv_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    logs_dir = runtime_root / 'logs'
    return RuntimeContext(runtime_root=runtime_root, repo_dir=repo_dir, venv_dir=venv_dir, python_exe=python_exe, logs_dir=logs_dir)


def _ensure_windows_binary_path() -> None:
    if os.name != 'nt':
        raise WorkspaceToolError('This runtime profile supports Windows + NVIDIA only.')


def _runtime_ready(runtime: RuntimeContext) -> bool:
    state = _load_state(runtime)
    return state.get('install_state') == 'ready' and runtime.python_exe.exists() and (runtime.repo_dir / 'run.py').exists()


def _prepare_repo(runtime: RuntimeContext) -> None:
    zip_url = os.environ.get('MODLY_UNIRIG_ZIP_URL', OFFICIAL_UNIRIG_ZIP_URL)
    cache_dir = runtime.runtime_root / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    if runtime.repo_dir.exists():
        shutil.rmtree(runtime.repo_dir, ignore_errors=True)

    zip_name = f'unirig-{OFFICIAL_UNIRIG_REF}.zip'
    zip_path = cache_dir / zip_name
    if not zip_path.exists():
        urllib.request.urlretrieve(zip_url, str(zip_path))

    unpack_root = runtime.runtime_root / 'repo_unpack'
    if unpack_root.exists():
        shutil.rmtree(unpack_root, ignore_errors=True)
    shutil.unpack_archive(str(zip_path), str(unpack_root))

    candidates = [p for p in unpack_root.iterdir() if p.is_dir()]
    if not candidates:
        raise WorkspaceToolError('Downloaded UniRig zip did not contain a repository directory')
    extracted_repo = candidates[0]
    shutil.move(str(extracted_repo), str(runtime.repo_dir))
    shutil.rmtree(unpack_root, ignore_errors=True)


def _resolve_python311_cmd() -> list[str]:
    env_bin = os.environ.get('MODLY_UNIRIG_PYTHON311_BIN', '').strip()
    if env_bin:
        return [env_bin]

    bundled_env = [
        'MODLY_BUNDLED_PYTHON311_BIN',
        'MODLY_EMBEDDED_PYTHON311_BIN',
    ]
    for key in bundled_env:
        value = os.environ.get(key, '').strip()
        if value:
            return [value]

    bundled_candidates = []
    if os.name == 'nt':
        bundled_candidates.extend([
            Path(os.environ.get('MODLY_APP_DIR', '')) / 'python' / 'python.exe',
            Path(os.environ.get('MODLY_HOME', '')) / 'python' / 'python.exe',
        ])
    for candidate in bundled_candidates:
        if str(candidate) and candidate.exists():
            return [str(candidate)]

    if shutil.which('py'):
        return ['py', '-3.11']
    if shutil.which('python3.11'):
        return ['python3.11']

    raise WorkspaceToolError(
        'Python 3.11 not found. Set MODLY_UNIRIG_PYTHON311_BIN or install Python 3.11.'
    )


def _create_venv(runtime: RuntimeContext) -> None:
    if runtime.python_exe.exists():
        return
    py311_cmd = _resolve_python311_cmd()
    _run([*py311_cmd, '-m', 'venv', str(runtime.venv_dir)], cwd=runtime.runtime_root)


def _install_requirements_excluding_flash_attn(runtime: RuntimeContext) -> None:
    req_path = runtime.repo_dir / 'requirements.txt'
    if not req_path.exists():
        raise WorkspaceToolError('UniRig requirements.txt not found')
    raw_lines = req_path.read_text(encoding='utf-8', errors='replace').splitlines()
    filtered: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        head = stripped.split(';', 1)[0].strip().split()[0].lower()
        if head.startswith('flash_attn') or head.startswith('flash-attn'):
            continue
        filtered.append(stripped)
    if not filtered:
        return
    _run([str(runtime.python_exe), '-m', 'pip', 'install', *filtered], cwd=runtime.repo_dir)


def _install_flash_attn(runtime: RuntimeContext) -> None:
    wheel_url = os.environ.get('MODLY_UNIRIG_FLASH_ATTN_WHEEL_URL', FLASH_ATTN_WHEEL_DEFAULT).strip() or FLASH_ATTN_WHEEL_DEFAULT
    cache_dir = runtime.runtime_root / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    wheel_name = wheel_url.rsplit('/', 1)[-1].split('?', 1)[0]
    wheel_path = cache_dir / wheel_name
    try:
        if not wheel_path.exists():
            urllib.request.urlretrieve(wheel_url, str(wheel_path))
        _run([str(runtime.python_exe), '-m', 'pip', 'install', str(wheel_path)], cwd=runtime.repo_dir)
    except Exception as exc:
        _append_flash_log(runtime, f'wheel_url={wheel_url}\nerror={exc}')
        raise WorkspaceToolError('flash-attn wheel install failed') from exc


def _append_flash_log(runtime: RuntimeContext, detail: str) -> None:
    runtime.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = runtime.logs_dir / 'flash_attn_install.log'
    with log_file.open('a', encoding='utf-8') as handle:
        handle.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}] {detail}\n')


def _missing_imports(runtime: RuntimeContext) -> list[str]:
    code = (
        'import importlib, json\n'
        f'mods={REQUIRED_IMPORTS!r}\n'
        'missing=[]\n'
        'for m in mods:\n'
        '  try:\n'
        '    importlib.import_module(m)\n'
        '  except Exception:\n'
        '    missing.append(m)\n'
        'print(json.dumps({"missing":missing}))\n'
    )
    output = _run_capture([str(runtime.python_exe), '-c', code], cwd=runtime.repo_dir)
    payload = json.loads(output.strip().splitlines()[-1])
    return payload.get('missing', [])


def _prepare_input_mesh(input_path: Path, temp_root: Path) -> Path:
    suffix = input_path.suffix.lower()
    if suffix in DIRECT_INPUT_SUFFIXES:
        return input_path.resolve()
    if suffix not in CONVERTIBLE_INPUT_SUFFIXES:
        raise WorkspaceToolError(f'Unsupported input format: {suffix}')
    converted_path = temp_root / f'{input_path.stem}_prepared.glb'
    loaded = trimesh.load(input_path, force='scene' if suffix == '.gltf' else None)
    scene = loaded.scene() if isinstance(loaded, trimesh.Trimesh) else loaded
    blob = scene.export(file_type='glb')
    converted_path.write_bytes(blob if isinstance(blob, (bytes, bytearray)) else bytes(blob))
    return converted_path


def _run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '').strip().splitlines()[-40:]
        raise WorkspaceToolError(f'Command failed: {" ".join(command)}\n' + '\n'.join(tail))


def _run_capture(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or '').strip().splitlines()[-40:]
        raise WorkspaceToolError(f'Command failed: {" ".join(command)}\n' + '\n'.join(tail))
    return result.stdout or result.stderr or ''


__all__ = ['UniRigWorkspaceTool']
