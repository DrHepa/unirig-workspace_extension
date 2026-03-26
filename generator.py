from __future__ import annotations

import json
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import trimesh

from services.workspace_tools_base import BaseWorkspaceTool, WorkspaceToolError


OFFICIAL_UNIRIG_ARCHIVE_URL = 'https://api.github.com/repos/VAST-AI-Research/UniRig/tarball/HEAD'
BOOTSTRAP_VERSION = 4
PYTHON_STANDALONE_VERSION = '3.11.9'
PYTHON_STANDALONE_RELEASE = '20240726'
DEFAULT_TORCH_SPEC = 'torch torchvision torchaudio'
DEFAULT_SPCONV_PACKAGE = 'spconv-cu120'
DEFAULT_PYG_PACKAGES = ('torch-scatter', 'torch-cluster')
DEFAULT_MIN_VRAM_GB = 8.0
INSTALL_STATES = {'not_installed', 'installing', 'ready', 'error'}
HEARTBEAT_INTERVAL_SEC = 5
PHASE_TIMEOUTS_SEC = {
    'creating venv': 900,
    'installing torch': 1800,
    'installing official requirements': 3600,
    'installing spconv/PyG': 3600,
    'validating CUDA': 900,
}

DIRECT_INPUT_SUFFIXES = {'.obj', '.fbx', '.glb', '.vrm'}
CONVERTIBLE_INPUT_SUFFIXES = {'.gltf', '.stl', '.ply'}
SUPPORTED_SUFFIXES = DIRECT_INPUT_SUFFIXES | CONVERTIBLE_INPUT_SUFFIXES

SKELETON_TASK = 'configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml'
SKIN_TASK = 'configs/task/quick_inference_unirig_skin.yaml'
SKIN_DATA_NAME = 'raw_data.npz'
MERGE_REQUIRE_SUFFIX = 'obj,fbx,FBX,dae,glb,gltf,vrm'
REQUIRED_IMPORTS: list[tuple[str, str]] = [
    ('lightning', 'import lightning as L'),
    ('pytorch_lightning', 'import pytorch_lightning'),
    ('transformers', 'import transformers'),
    ('box', 'import box'),
    ('einops', 'import einops'),
    ('omegaconf', 'import omegaconf'),
    ('timm', 'import timm'),
    ('trimesh', 'import trimesh'),
    ('open3d', 'import open3d'),
    ('pyrender', 'import pyrender'),
    ('huggingface_hub', 'import huggingface_hub'),
    ('wandb', 'import wandb'),
    ('bpy', 'import bpy'),
    ('torch', 'import torch'),
    ('torch_scatter', 'import torch_scatter'),
    ('torch_cluster', 'import torch_cluster'),
    ('spconv', 'import spconv'),
]


@dataclass(frozen=True)
class RuntimeContext:
    runtime_root: Path
    repo_dir: Path
    venv_dir: Path
    python_exe: Path
    logs_dir: Path
    external_repo: bool = False


@dataclass(frozen=True)
class UniRigStageCommands:
    skeleton: list[str]
    skin: list[str]
    merge: list[str]


@dataclass(frozen=True)
class RuntimeProfile:
    profile_id: str
    python_minor: str
    torch: str
    torchvision: str
    torchaudio: str
    torch_index_url: str
    pyg_wheel_url: str
    torch_scatter: str
    torch_cluster: str
    spconv_package: str
    cuda_flavor: str


class UniRigWorkspaceTool(BaseWorkspaceTool):
    TOOL_ID = 'unirig-workspace-v1'
    DISPLAY_NAME = 'UniRig Workspace Tool'
    TOOL_KIND = 'mesh_rigger'
    PRIORITY = 100

    def __init__(self, workspace_dir: Path) -> None:
        super().__init__(workspace_dir)
        self._runtime_ctx: RuntimeContext | None = None

    def unload(self) -> None:
        self._runtime_ctx = None
        super().unload()

    def runtime_status(self) -> dict:
        runtime = _resolve_runtime_context()
        state = _load_bootstrap_state(runtime)
        if not state:
            state = _default_bootstrap_state(runtime)
        state = _normalize_state(state, runtime)

        if state.get('install_state') == 'ready' and not _runtime_ready(runtime):
            state['install_state'] = 'error'
            state['last_error'] = 'Runtime state says ready, but runtime files are missing or invalid.'
        return state

    def install_runtime(self, progress_cb: Optional[Callable[[int, str], None]] = None) -> dict:
        runtime = _resolve_runtime_context()
        force_bootstrap = _env_truthy('MODLY_UNIRIG_FORCE_BOOTSTRAP', False)
        current = _normalize_state(_load_bootstrap_state(runtime) or _default_bootstrap_state(runtime), runtime)
        if current['install_state'] == 'ready' and _runtime_ready(runtime) and not force_bootstrap:
            self._runtime_ctx = runtime
            return current

        runtime.runtime_root.mkdir(parents=True, exist_ok=True)
        runtime.logs_dir.mkdir(parents=True, exist_ok=True)
        bootstrap_log = runtime.logs_dir / f'bootstrap_{int(time.time())}.log'

        if force_bootstrap:
            if runtime.venv_dir.exists():
                shutil.rmtree(runtime.venv_dir, ignore_errors=True)
            if runtime.repo_dir.exists() and not runtime.external_repo:
                shutil.rmtree(runtime.repo_dir, ignore_errors=True)
            markers = runtime.runtime_root / '.markers'
            if markers.exists():
                shutil.rmtree(markers, ignore_errors=True)

        state = _default_bootstrap_state(runtime)
        state.update(
            {
                'install_state': 'installing',
                'started_at': int(time.time()),
                'step': 'init',
                'percent': 0,
                'bootstrap_log': str(bootstrap_log),
            }
        )
        _write_bootstrap_state(runtime, state)
        _log_bootstrap_event(bootstrap_log, 'install_runtime', 'start', 'bootstrap started')

        try:
            _set_state_and_report(runtime, state, progress_cb, 5, 'preparing runtime directories')
            _set_state_and_report(runtime, state, progress_cb, 12, 'finding python 3.11')
            python_resolution = _resolve_python311_command(
                runtime,
                state,
                phase_cb=lambda percent, step: _set_state_and_report(runtime, state, progress_cb, percent, step),
                bootstrap_log=bootstrap_log,
            )
            python_cmd = python_resolution['command']
            _set_state_and_report(
                runtime,
                state,
                progress_cb,
                30,
                'python 3.11 selected',
                selected_python=python_resolution['selected_python'],
                selected_python_version=python_resolution['selected_python_version'],
                selected_python_source=python_resolution['selected_python_source'],
                standalone_python_root=python_resolution.get('standalone_python_root', ''),
                python_lookup_attempts=python_resolution['attempts'],
                install_phases=python_resolution['phases'],
            )

            _set_state_and_report(runtime, state, progress_cb, 35, 'preparing UniRig repo')
            if not _phase_marked(runtime, 'prepare_repo'):
                self._prepare_repo(
                    runtime,
                    bootstrap_log=bootstrap_log,
                    heartbeat_cb=lambda: _heartbeat_state(runtime, state, progress_cb),
                    phase_cb=lambda percent, step: _set_state_and_report(runtime, state, progress_cb, percent, step),
                )
                _mark_phase(runtime, 'prepare_repo')
            _complete_phase(state, 'prepare_repo')
            _append_install_phase(state, 'prepare_repo', 'ok')
            _write_bootstrap_state(runtime, state)

            _set_state_and_report(runtime, state, progress_cb, 55, 'creating venv')
            if _phase_marked(runtime, 'create_venv') and _is_venv_valid(runtime):
                _log_bootstrap_event(bootstrap_log, 'creating venv', 'ok', 'skipped: existing venv is valid')
            else:
                _run_subprocess_with_heartbeat(
                    runtime,
                    state,
                    progress_cb,
                    [*python_cmd, '-m', 'venv', str(runtime.venv_dir)],
                    cwd=runtime.runtime_root,
                    log_path=bootstrap_log,
                    phase='creating venv',
                    timeout_sec=PHASE_TIMEOUTS_SEC['creating venv'],
                )
                if not runtime.python_exe.exists():
                    raise WorkspaceToolError(f'Venv creation succeeded but Python executable was not found: {runtime.python_exe}')
                _mark_phase(runtime, 'create_venv')
            _complete_phase(state, 'create_venv')
            _append_install_phase(state, 'create_venv', 'ok')
            _write_bootstrap_state(runtime, state)

            _set_state_and_report(runtime, state, progress_cb, 65, 'installing torch')
            _bootstrap_runtime(
                runtime,
                state,
                bootstrap_log,
                progress_cb=progress_cb,
                phase_cb=lambda percent, step: _set_state_and_report(runtime, state, progress_cb, percent, step),
            )
            _append_install_phase(state, 'bootstrap_runtime', 'ok')
            _write_bootstrap_state(runtime, state)

            _set_state_and_report(
                runtime,
                state,
                progress_cb,
                92,
                'validating required imports',
                phase_timeout_sec=PHASE_TIMEOUTS_SEC['validating CUDA'],
                phase_started_at=int(time.time()),
            )
            validation = _validate_runtime(runtime, bootstrap_log=bootstrap_log)
            _complete_phase(state, 'validate_cuda')
            _append_install_phase(state, 'validate', 'ok')
            state['missing_required_modules'] = validation.get('missing_required_modules', [])
            state['runpy_smoke_ok'] = bool(validation.get('runpy_smoke_ok'))
            state['required_baseline_installed'] = bool(validation.get('baseline_installed'))
            state['last_validation_error'] = ''
            _write_bootstrap_state(runtime, state)

            _set_state_and_report(
                runtime,
                state,
                progress_cb,
                100,
                'ready',
                install_state='ready',
                completed_at=int(time.time()),
                last_error='',
                subprocess_alive=False,
                subprocess_pid=None,
                subprocess_cmd='',
                python_exe=str(runtime.python_exe),
            )
            state['validation'] = validation
            _write_bootstrap_state(runtime, state)
            self._runtime_ctx = runtime
            _log_bootstrap_event(bootstrap_log, 'install_runtime', 'ok', 'runtime ready')
            return state
        except Exception as exc:
            state['last_validation_error'] = str(exc)
            if not state.get('last_error_excerpt') and state.get('bootstrap_log'):
                state['last_error_excerpt'] = _tail_text(Path(str(state['bootstrap_log'])), 80)
            _set_state_and_report(
                runtime,
                state,
                progress_cb,
                int(state.get('percent', 0)),
                'failed',
                install_state='error',
                completed_at=int(time.time()),
                last_error=str(exc),
                subprocess_alive=False,
            )
            _append_install_phase(state, state.get('step', 'failed'), 'error', str(exc))
            _write_bootstrap_state(runtime, state)
            _log_bootstrap_event(bootstrap_log, state.get('step', 'failed'), 'error', str(exc))
            raise WorkspaceToolError(f'Failed to install UniRig runtime: {exc}') from exc

    def uninstall_runtime(self) -> None:
        runtime = _resolve_runtime_context()
        self._runtime_ctx = None
        if runtime.venv_dir.exists():
            shutil.rmtree(runtime.venv_dir, ignore_errors=True)
        if runtime.repo_dir.exists() and not runtime.external_repo:
            shutil.rmtree(runtime.repo_dir, ignore_errors=True)
        state = _default_bootstrap_state(runtime)
        _write_bootstrap_state(runtime, state)

    def validate_runtime(self) -> dict:
        runtime = _resolve_runtime_context()
        status = self.runtime_status()
        if status.get('install_state') != 'ready':
            return {'ok': False, 'reason': f"install_state={status.get('install_state')}"}
        try:
            payload = _validate_runtime(runtime)
            return {'ok': True, 'details': payload}
        except Exception as exc:
            return {'ok': False, 'reason': str(exc)}

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        suffix = input_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise WorkspaceToolError(
                f'Unsupported mesh format: {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}'
            )

        status = self.runtime_status()
        if status.get('install_state') != 'ready':
            raise WorkspaceToolError(
                'UniRig runtime is not ready. Install it first from the Runtime panel. '
                f"Current state: {status.get('install_state')} ({status.get('last_error') or 'no details'})"
            )

        seed = _safe_int(params.get('seed'), 12345)
        runtime = self._runtime_ctx or _resolve_runtime_context()
        if not _runtime_ready(runtime):
            raise WorkspaceToolError('UniRig runtime files are missing. Reinstall runtime from the Runtime panel.')

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        output_path = output_dir / f'{input_path.stem}_unirig_{timestamp}.glb'
        meta_path = output_dir / f'{output_path.stem}.rigmeta.json'
        run_log_dir = runtime.logs_dir / f'rig_run_{timestamp}'
        run_log_dir.mkdir(parents=True, exist_ok=True)

        self._report(progress_cb, 35, 'Preparing workspace mesh…')
        with tempfile.TemporaryDirectory(prefix='modly_unirig_stage_') as tmp_dir:
            stage_root = Path(tmp_dir)
            prepared_input = _prepare_input_mesh(input_path, stage_root)
            skeleton_output = stage_root / 'skeleton_stage.fbx'
            skin_output = stage_root / 'skin_stage.fbx'
            skeleton_npz_dir = stage_root / 'skeleton_npz'
            skin_npz_dir = stage_root / 'skin_npz'

            commands = build_unirig_commands(
                python_exe=runtime.python_exe,
                prepared_input=prepared_input,
                skeleton_output=skeleton_output,
                skin_output=skin_output,
                final_output=output_path,
                skeleton_npz_dir=skeleton_npz_dir,
                skin_npz_dir=skin_npz_dir,
                seed=seed,
            )

            self._report(progress_cb, 50, 'Predicting skeleton…')
            _run_logged(commands.skeleton, cwd=runtime.repo_dir, log_path=run_log_dir / '01_skeleton.log')
            if not skeleton_output.exists():
                raise WorkspaceToolError('UniRig skeleton stage completed without producing a skeleton FBX.')

            self._report(progress_cb, 72, 'Predicting skin weights…')
            _run_logged(commands.skin, cwd=runtime.repo_dir, log_path=run_log_dir / '02_skin.log')
            if not skin_output.exists():
                raise WorkspaceToolError('UniRig skin stage completed without producing a skinned FBX.')

            self._report(progress_cb, 90, 'Merging rig back into the original mesh…')
            _run_logged(commands.merge, cwd=runtime.repo_dir, log_path=run_log_dir / '03_merge.log')

        if not output_path.exists():
            raise WorkspaceToolError('UniRig merge finished but no output GLB was produced.')

        runtime_info = _query_runtime_info(runtime)
        meta = {
            'tool_id': self.TOOL_ID,
            'tool_name': self.DISPLAY_NAME,
            'rig_style': 'unirig_workspace_v1',
            'source_mesh': input_path.name,
            'input_format': input_path.suffix.lower(),
            'output_mesh': output_path.name,
            'output_format': '.glb',
            'created_at': timestamp,
            'seed': seed,
            'runtime': {
                'runtime_root': str(runtime.runtime_root),
                'repo_dir': str(runtime.repo_dir),
                'venv_dir': str(runtime.venv_dir),
                'external_repo': runtime.external_repo,
                'python_exe': str(runtime.python_exe),
                'gpu': runtime_info,
            },
            'logs': {
                'skeleton': str(run_log_dir / '01_skeleton.log'),
                'skin': str(run_log_dir / '02_skin.log'),
                'merge': str(run_log_dir / '03_merge.log'),
            },
            'pipeline': [
                'skeleton_prediction',
                'skin_prediction',
                'merge_to_original_mesh',
            ],
            'notes': [
                'Rig produced with an isolated local UniRig runtime.',
                'The persistent workspace output is a rigged GLB. Intermediate FBX stage files are temporary.',
            ],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

        self._report(progress_cb, 100, 'Rig ready')
        return output_path

    def _prepare_repo(
        self,
        runtime: RuntimeContext,
        bootstrap_log: Path | None = None,
        heartbeat_cb: Optional[Callable[[], None]] = None,
        phase_cb: Optional[Callable[[int, str], None]] = None,
    ) -> bool:
        if runtime.repo_dir.exists() and (runtime.repo_dir / 'run.py').exists():
            return False
        if runtime.external_repo:
            raise WorkspaceToolError(f'UniRig repo override is invalid: {runtime.repo_dir}')
        _download_unirig_repo(
            runtime,
            bootstrap_log=bootstrap_log,
            heartbeat_cb=heartbeat_cb,
            phase_cb=phase_cb,
        )
        return True

    @staticmethod
    def _report(progress_cb: Optional[Callable[[int, str], None]], pct: int, message: str) -> None:
        if progress_cb:
            progress_cb(pct, message)


def build_unirig_commands(
    python_exe: Path,
    prepared_input: Path,
    skeleton_output: Path,
    skin_output: Path,
    final_output: Path,
    skeleton_npz_dir: Path,
    skin_npz_dir: Path,
    seed: int,
) -> UniRigStageCommands:
    skeleton_cmd = [
        str(python_exe),
        'run.py',
        f'--task={SKELETON_TASK}',
        f'--seed={seed}',
        f'--input={prepared_input}',
        f'--output={skeleton_output}',
        f'--npz_dir={skeleton_npz_dir}',
    ]
    skin_cmd = [
        str(python_exe),
        'run.py',
        f'--task={SKIN_TASK}',
        f'--seed={seed}',
        f'--input={skeleton_output}',
        f'--output={skin_output}',
        f'--npz_dir={skin_npz_dir}',
        f'--data_name={SKIN_DATA_NAME}',
    ]
    merge_cmd = [
        str(python_exe),
        '-m',
        'src.inference.merge',
        f'--require_suffix={MERGE_REQUIRE_SUFFIX}',
        '--num_runs=1',
        '--id=0',
        f'--source={skin_output}',
        f'--target={prepared_input}',
        f'--output={final_output}',
    ]
    return UniRigStageCommands(skeleton=skeleton_cmd, skin=skin_cmd, merge=merge_cmd)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _read_modly_settings() -> dict:
    user_data_dir = os.environ.get('MODLY_USERDATA_DIR')
    if not user_data_dir:
        return {}
    settings_path = Path(user_data_dir) / 'settings.json'
    if not settings_path.exists():
        return {}
    try:
        return json.loads(settings_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _resolve_runtime_context() -> RuntimeContext:
    override_root = os.environ.get('MODLY_UNIRIG_RUNTIME_DIR')
    if override_root:
        runtime_root = Path(override_root).expanduser().resolve()
    else:
        settings = _read_modly_settings()
        deps_dir = settings.get('dependenciesDir')
        if deps_dir:
            runtime_root = Path(deps_dir).expanduser().resolve() / 'unirig-workspace-v1'
        else:
            runtime_root = Path(__file__).resolve().parent / '_runtime'

    repo_override = os.environ.get('MODLY_UNIRIG_REPO_DIR')
    if repo_override:
        repo_dir = Path(repo_override).expanduser().resolve()
        external_repo = True
    else:
        repo_dir = runtime_root / 'repo'
        external_repo = False

    venv_dir = runtime_root / 'venv'
    python_exe = _venv_python(venv_dir)
    logs_dir = runtime_root / 'logs'
    return RuntimeContext(
        runtime_root=runtime_root,
        repo_dir=repo_dir,
        venv_dir=venv_dir,
        python_exe=python_exe,
        logs_dir=logs_dir,
        external_repo=external_repo,
    )


def _venv_python(venv_dir: Path) -> Path:
    if os.name == 'nt':
        return venv_dir / 'Scripts' / 'python.exe'
    return venv_dir / 'bin' / 'python'


def _runtime_ready(runtime: RuntimeContext) -> bool:
    state = _load_bootstrap_state(runtime)
    return (
        runtime.python_exe.exists()
        and (runtime.repo_dir / 'run.py').exists()
        and state.get('install_state') == 'ready'
    )


def get_supported_runtime_profiles() -> list[RuntimeProfile]:
    return [
        RuntimeProfile(
            profile_id='win-cu128-stable',
            python_minor='3.11',
            torch='2.7.1+cu128',
            torchvision='0.22.1+cu128',
            torchaudio='2.7.1+cu128',
            torch_index_url='https://download.pytorch.org/whl/cu128',
            pyg_wheel_url='https://data.pyg.org/whl/torch-2.7.1+cu128.html',
            torch_scatter='2.1.2+pt27cu128',
            torch_cluster='1.6.3+pt27cu128',
            spconv_package='spconv-cu120',
            cuda_flavor='cu128',
        ),
        RuntimeProfile(
            profile_id='win-cu126-stable',
            python_minor='3.11',
            torch='2.7.1+cu126',
            torchvision='0.22.1+cu126',
            torchaudio='2.7.1+cu126',
            torch_index_url='https://download.pytorch.org/whl/cu126',
            pyg_wheel_url='https://data.pyg.org/whl/torch-2.7.1+cu126.html',
            torch_scatter='2.1.2+pt27cu126',
            torch_cluster='1.6.3+pt27cu126',
            spconv_package='spconv-cu120',
            cuda_flavor='cu126',
        ),
    ]


def resolve_runtime_profile(runtime: RuntimeContext) -> tuple[RuntimeProfile, dict]:
    os_name = platform.system().lower()
    gpu_info = _detect_cuda_environment(runtime)
    forced_profile_id = os.environ.get('MODLY_UNIRIG_RUNTIME_PROFILE', '').strip()
    if os_name != 'windows':
        raise WorkspaceToolError('UniRig runtime profile resolver currently supports CUDA bootstrap on Windows only.')
    if not forced_profile_id and not gpu_info.get('has_nvidia_gpu'):
        raise WorkspaceToolError(
            'UniRig runtime requires a CUDA-enabled GPU profile on Windows. '
            'CPU torch profile is not used by default for this runtime.'
        )
    profiles = get_supported_runtime_profiles()
    if forced_profile_id:
        forced_profile = next((p for p in profiles if p.profile_id == forced_profile_id), None)
        if not forced_profile:
            raise WorkspaceToolError(f'Unknown runtime profile override: {forced_profile_id}')
        return forced_profile, {'reason': f'forced profile override: {forced_profile_id}', 'gpu': gpu_info, 'wheel_check': {'ok': True}}
    reasons: list[str] = []
    for profile in profiles:
        availability = _validate_profile_binary_wheels(profile)
        if availability['ok']:
            reason = (
                f"selected {profile.profile_id}: NVIDIA GPU detected"
                f"{' via nvidia-smi' if gpu_info.get('nvidia_smi_ok') else ''}; binary PyG wheels verified"
            )
            return profile, {'reason': reason, 'gpu': gpu_info, 'wheel_check': availability}
        reasons.append(f"{profile.profile_id} rejected: {availability['reason']}")
    raise WorkspaceToolError(
        'No compatible Windows CUDA runtime profile has complete binary wheels for cp311 win_amd64. '
        f'Checked profiles: {"; ".join(reasons)}'
    )


def _detect_cuda_environment(runtime: RuntimeContext) -> dict:
    forced = os.environ.get('MODLY_UNIRIG_RUNTIME_PROFILE', '').strip()
    if forced:
        return {'has_nvidia_gpu': True, 'nvidia_smi_ok': False, 'forced_profile': forced}
    try:
        output = _run_capture(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], cwd=runtime.runtime_root)
        line = next((ln.strip() for ln in output.splitlines() if ln.strip()), '')
        return {'has_nvidia_gpu': bool(line), 'nvidia_smi_ok': bool(line), 'nvidia_smi': line}
    except Exception as exc:
        if _env_truthy('MODLY_UNIRIG_ASSUME_NVIDIA_GPU', False):
            return {'has_nvidia_gpu': True, 'nvidia_smi_ok': False, 'warning': f'nvidia-smi check failed: {exc}'}
        return {'has_nvidia_gpu': False, 'nvidia_smi_ok': False, 'warning': f'nvidia-smi check failed: {exc}'}


def _validate_profile_binary_wheels(profile: RuntimeProfile) -> dict:
    if os.environ.get('MODLY_UNIRIG_SKIP_WHEEL_PROBE') == '1':
        return {'ok': True, 'reason': 'probe skipped by override'}
    try:
        request = urllib.request.Request(profile.pyg_wheel_url, headers={'User-Agent': 'Modly-UniRig-WheelProbe'})
        with urllib.request.urlopen(request, timeout=20) as response:
            html = response.read().decode('utf-8', errors='ignore')
    except Exception as exc:
        return {'ok': False, 'reason': f'failed to fetch PyG wheel index: {exc}'}
    required_tags = [profile.torch_scatter, profile.torch_cluster]
    for tag in required_tags:
        if tag not in html:
            return {'ok': False, 'reason': f'missing package tag {tag} in wheel index'}
    win311_pattern = re.compile(r'cp311-cp311-win_amd64', re.IGNORECASE)
    if not win311_pattern.search(html):
        return {'ok': False, 'reason': 'missing cp311 win_amd64 wheels in wheel index'}
    return {'ok': True, 'reason': 'binary wheel matrix present'}


def _bootstrap_state_path(runtime: RuntimeContext) -> Path:
    return runtime.runtime_root / 'bootstrap_state.json'


def _default_bootstrap_state(runtime: RuntimeContext) -> dict:
    now = int(time.time())
    return {
        'bootstrap_version': BOOTSTRAP_VERSION,
        'install_state': 'not_installed',
        'percent': 0,
        'step': 'idle',
        'message': 'not installed',
        'last_error': '',
        'started_at': None,
        'completed_at': None,
        'updated_at': now,
        'last_heartbeat_at': now,
        'phase_started_at': None,
        'last_output_at': None,
        'subprocess_pid': None,
        'subprocess_cmd': '',
        'subprocess_alive': False,
        'phase_timeout_sec': None,
        'last_error_excerpt': '',
        'bootstrap_log': '',
        'python_exe': str(runtime.python_exe),
        'repo_dir': str(runtime.repo_dir),
        'venv_dir': str(runtime.venv_dir),
        'runtime_root': str(runtime.runtime_root),
        'selected_python': '',
        'selected_python_version': '',
        'selected_python_source': '',
        'python_lookup_attempts': [],
        'standalone_python_root': '',
        'selected_runtime_profile': '',
        'selected_torch_version': '',
        'selected_cuda_flavor': '',
        'selected_pyg_wheel_url': '',
        'selected_torch_index_url': '',
        'binary_only_mode': True,
        'source_build_allowed': _env_truthy('MODLY_UNIRIG_ALLOW_SOURCE_BUILDS', False),
        'required_baseline_source': '',
        'required_baseline_installed': False,
        'missing_required_modules': [],
        'runpy_smoke_ok': False,
        'last_validation_error': '',
        'install_phases': [],
        'completed_phases': [],
    }


def _normalize_state(state: dict, runtime: RuntimeContext) -> dict:
    normalized = _default_bootstrap_state(runtime)
    normalized.update(state)
    if normalized.get('install_state') not in INSTALL_STATES:
        normalized['install_state'] = 'error'
        normalized['last_error'] = f"Unknown install_state value: {normalized.get('install_state')}"
    return normalized


def _load_bootstrap_state(runtime: RuntimeContext) -> dict:
    state_path = _bootstrap_state_path(runtime)
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_bootstrap_state(runtime: RuntimeContext, data: dict) -> None:
    runtime.runtime_root.mkdir(parents=True, exist_ok=True)
    _bootstrap_state_path(runtime).write_text(json.dumps(data, indent=2), encoding='utf-8')


def _set_state(runtime: RuntimeContext, state: dict, **updates: object) -> None:
    state.update(updates)
    _write_bootstrap_state(runtime, state)


def _set_state_and_report(
    runtime: RuntimeContext,
    state: dict,
    progress_cb: Optional[Callable[[int, str], None]],
    percent: int,
    step: str,
    **updates: object,
) -> None:
    now = int(time.time())
    message = updates.pop('message', step)
    phase_started_at = updates.pop('phase_started_at', state.get('phase_started_at'))
    state.update(
        {
            'percent': percent,
            'step': step,
            'message': message,
            'updated_at': now,
            'last_heartbeat_at': now,
            'phase_started_at': phase_started_at if phase_started_at is not None else now,
            **updates,
        }
    )
    _write_bootstrap_state(runtime, state)
    if progress_cb:
        progress_cb(percent, step)


def _heartbeat_state(
    runtime: RuntimeContext,
    state: dict,
    progress_cb: Optional[Callable[[int, str], None]],
) -> None:
    now = int(time.time())
    state.update(
        {
            'updated_at': now,
            'last_heartbeat_at': now,
            'message': state.get('message') or state.get('step', 'installing'),
        }
    )
    _write_bootstrap_state(runtime, state)


def _complete_phase(state: dict, phase_name: str) -> None:
    completed = state.setdefault('completed_phases', [])
    if isinstance(completed, list) and phase_name not in completed:
        completed.append(phase_name)


def _append_install_phase(state: dict, phase: str, status: str, detail: str = '') -> None:
    phases = state.setdefault('install_phases', [])
    if isinstance(phases, list):
        phases.append({'phase': phase, 'status': status, 'detail': detail, 'at': int(time.time())})


def _download_unirig_repo(
    runtime: RuntimeContext,
    bootstrap_log: Path | None = None,
    heartbeat_cb: Optional[Callable[[], None]] = None,
    phase_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    archive_url = os.environ.get('MODLY_UNIRIG_REPO_ARCHIVE_URL', OFFICIAL_UNIRIG_ARCHIVE_URL)
    tmp_dir = runtime.runtime_root / '_repo_download'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / 'unirig.tar.gz'

    try:
        if phase_cb:
            phase_cb(42, 'downloading UniRig repo')
        if bootstrap_log:
            _log_bootstrap_event(bootstrap_log, 'downloading UniRig repo', 'start', archive_url)
        _download_file(archive_url, archive_path, heartbeat_cb=heartbeat_cb)
        if bootstrap_log:
            _log_bootstrap_event(bootstrap_log, 'downloading UniRig repo', 'ok')
    except Exception as exc:
        raise WorkspaceToolError(
            'Failed to download the official UniRig repository. '
            'Set MODLY_UNIRIG_REPO_DIR to an already downloaded local checkout if you want a fully offline bootstrap. '
            f'Original error: {exc}'
        ) from exc

    extract_dir = tmp_dir / 'extract'
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        if phase_cb:
            phase_cb(48, 'extracting UniRig repo')
        if bootstrap_log:
            _log_bootstrap_event(bootstrap_log, 'extracting UniRig repo', 'start', str(archive_path))
        _extract_archive(archive_path, extract_dir, heartbeat_cb=heartbeat_cb)
        if bootstrap_log:
            _log_bootstrap_event(bootstrap_log, 'extracting UniRig repo', 'ok')
    except Exception as exc:
        raise WorkspaceToolError(f'Failed to extract UniRig repository archive: {exc}') from exc

    candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise WorkspaceToolError('Unexpected UniRig archive layout while extracting repository.')

    if runtime.repo_dir.exists():
        shutil.rmtree(runtime.repo_dir, ignore_errors=True)
    shutil.copytree(candidates[0], runtime.repo_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_python311_command(
    runtime: RuntimeContext,
    state: dict | None = None,
    phase_cb: Optional[Callable[[int, str], None]] = None,
    bootstrap_log: Path | None = None,
) -> dict:
    settings = _read_modly_settings()
    attempts: list[dict] = []
    phases: list[dict] = []

    def try_candidate(command: list[str], source: str) -> tuple[bool, dict]:
        info = _probe_python_version(command)
        attempt = {'source': source, 'command': _format_command(command), 'ok': False, 'version': info.get('version', '')}
        if not info['ok']:
            attempt['reason'] = info['reason']
            attempts.append(attempt)
            return False, info
        if info['major'] == 3 and info['minor'] == 11:
            attempt['ok'] = True
            attempts.append(attempt)
            return True, info
        attempt['reason'] = (
            f"Rejected version {info['version']}. UniRig requires standalone Python 3.11. "
            'Blender Python is not used for this bootstrap.'
        )
        attempts.append(attempt)
        return False, info

    env_python = os.environ.get('MODLY_UNIRIG_PYTHON311_BIN')
    if env_python:
        cmd = _split_command(env_python)
        ok, info = try_candidate(cmd, 'env_override')
        if ok:
            return {
                'command': cmd,
                'selected_python': cmd[0],
                'selected_python_version': info['version'],
                'selected_python_source': 'env_override',
                'attempts': attempts,
                'phases': phases,
            }

    for key in ('externalPython311Bin', 'python311Bin', 'external_python_311_bin'):
        candidate = settings.get(key)
        if isinstance(candidate, str) and candidate.strip():
            cmd = _split_command(candidate)
            ok, info = try_candidate(cmd, 'modly_settings')
            if ok:
                return {
                    'command': cmd,
                    'selected_python': cmd[0],
                    'selected_python_version': info['version'],
                    'selected_python_source': 'modly_settings',
                    'attempts': attempts,
                    'phases': phases,
                }

    bundled = _find_modly_bundled_python()
    if bundled:
        if phase_cb:
            phase_cb(18, 'reusing bundled python')
        cmd = [str(bundled)]
        ok, info = try_candidate(cmd, 'modly_bundled_python')
        if ok:
            return {
                'command': cmd,
                'selected_python': str(bundled),
                'selected_python_version': info['version'],
                'selected_python_source': 'modly_bundled_python',
                'attempts': attempts,
                'phases': phases,
            }

    local_python = _ensure_runtime_python311(
        runtime,
        state=state,
        phases=phases,
        phase_cb=phase_cb,
        bootstrap_log=bootstrap_log,
    )
    if local_python:
        cmd = [str(local_python)]
        ok, info = try_candidate(cmd, 'runtime_local_python')
        if ok:
            return {
                'command': cmd,
                'selected_python': str(local_python),
                'selected_python_version': info['version'],
                'selected_python_source': 'runtime_local_python',
                'attempts': attempts,
                'standalone_python_root': str(runtime.runtime_root / 'python311'),
                'phases': phases,
            }

    if os.name == 'nt' and shutil.which('py'):
        cmd = ['py', '-3.11']
        ok, info = try_candidate(cmd, 'py_launcher')
        if ok:
            return {
                'command': cmd,
                'selected_python': 'py -3.11',
                'selected_python_version': info['version'],
                'selected_python_source': 'py_launcher',
                'attempts': attempts,
                'phases': phases,
            }

    for candidate in ('python3.11', 'python'):
        resolved = shutil.which(candidate)
        if not resolved:
            continue
        cmd = [resolved]
        ok, info = try_candidate(cmd, 'PATH_python')
        if ok:
            return {
                'command': cmd,
                'selected_python': resolved,
                'selected_python_version': info['version'],
                'selected_python_source': 'PATH_python',
                'attempts': attempts,
                'phases': phases,
            }

    rejection_log = '\n'.join(
        f"- {a['source']}: {a['command']} -> {a.get('reason', 'accepted')}" for a in attempts
    )
    raise WorkspaceToolError(
        'UniRig requires standalone Python 3.11. Blender Python is not used for this bootstrap.\n'
        f'Lookup attempts:\n{rejection_log or "- (no candidates found)"}'
    )


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise WorkspaceToolError('Python 3.11 command is empty.')
    return parts


def _is_python311(command: Sequence[str]) -> bool:
    info = _probe_python_version(command)
    return bool(info['ok'] and info['major'] == 3 and info['minor'] == 11)


def _probe_python_version(command: Sequence[str]) -> dict:
    try:
        result = subprocess.run(
            [
                *command,
                '-c',
                'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")',
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return {'ok': False, 'reason': f'Failed to execute candidate: {exc}'}
    if result.returncode != 0:
        tail = '\n'.join((result.stderr or result.stdout).splitlines()[-5:])
        return {'ok': False, 'reason': f'Execution failed (exit {result.returncode}): {tail}'}
    version = (result.stdout or '').strip()
    parts = version.split('.')
    if len(parts) != 3:
        return {'ok': False, 'reason': f'Unexpected version output: {version!r}', 'version': version}
    try:
        major, minor, micro = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return {'ok': False, 'reason': f'Invalid version output: {version!r}', 'version': version}
    return {'ok': True, 'version': version, 'major': major, 'minor': minor, 'micro': micro}


def _find_modly_bundled_python() -> Path | None:
    env_roots = [
        os.environ.get('MODLY_RESOURCES_DIR'),
        os.environ.get('MODLY_APP_DIR'),
        os.environ.get('MODLY_BACKEND_DIR'),
    ]
    candidate_roots: list[Path] = []
    for raw in env_roots:
        if raw:
            candidate_roots.append(Path(raw).expanduser())

    file_root = Path(__file__).resolve()
    candidate_roots.extend(
        [
            file_root.parent,
            *file_root.parents[:5],
            Path(sys.executable).resolve().parent,
        ]
    )

    checked: set[str] = set()
    for root in candidate_roots:
        for probe in _modly_python_candidates_from_root(root):
            key = str(probe.resolve()) if probe.exists() else str(probe)
            if key in checked:
                continue
            checked.add(key)
            if probe.exists() and _is_python311([str(probe)]):
                return probe
    return None


def _modly_python_candidates_from_root(root: Path) -> list[Path]:
    resources_candidates = [
        root / 'resources',
        root.parent / 'resources',
    ]
    out: list[Path] = []
    for resources in resources_candidates:
        embed = resources / 'python-embed'
        if os.name == 'nt':
            out.append(embed / 'python.exe')
        else:
            out.extend([embed / 'bin' / 'python3.11', embed / 'bin' / 'python3'])
    return out


def _ensure_runtime_python311(
    runtime: RuntimeContext,
    state: dict | None,
    phases: list[dict],
    phase_cb: Optional[Callable[[int, str], None]] = None,
    bootstrap_log: Path | None = None,
) -> Path | None:
    root = runtime.runtime_root / 'python311'
    existing = _find_python_in_tree(root)
    if existing and _is_python311([str(existing)]):
        if phase_cb:
            phase_cb(22, 'reusing local python 3.11')
        phases.append({'phase': 'python_standalone', 'status': 'reuse', 'detail': str(existing), 'at': int(time.time())})
        return existing

    archive_override = os.environ.get('MODLY_UNIRIG_PYTHON_STANDALONE_ARCHIVE')
    url_override = os.environ.get('MODLY_UNIRIG_PYTHON_STANDALONE_URL')
    archive_path = Path(archive_override).expanduser() if archive_override else None
    download_url = url_override or _default_python_standalone_url()
    if not archive_path and not download_url:
        return None

    phases.append({'phase': 'python_standalone', 'status': 'start', 'detail': str(root), 'at': int(time.time())})
    if state is not None:
        state['standalone_python_root'] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    archive_file = root / 'python_standalone_archive'
    if archive_path:
        if phase_cb:
            phase_cb(22, 'downloading local python 3.11')
        shutil.copy2(archive_path, archive_file)
    else:
        if phase_cb:
            phase_cb(22, 'downloading local python 3.11')
        _download_file(download_url, archive_file, heartbeat_cb=(lambda: phase_cb(22, 'downloading local python 3.11')) if phase_cb else None)
    if phase_cb:
        phase_cb(28, 'extracting local python 3.11')
    _extract_archive(
        archive_file,
        root,
        heartbeat_cb=(lambda: phase_cb(28, 'extracting local python 3.11')) if phase_cb else None,
    )
    archive_file.unlink(missing_ok=True)

    resolved = _find_python_in_tree(root)
    if not resolved:
        phases.append({'phase': 'python_standalone', 'status': 'error', 'detail': 'No python binary found after extraction', 'at': int(time.time())})
        return None
    phases.append({'phase': 'python_standalone', 'status': 'ok', 'detail': str(resolved), 'at': int(time.time())})
    return resolved


def _default_python_standalone_url() -> str | None:
    machine = platform.machine().lower()
    arch = 'x86_64' if machine in {'x86_64', 'amd64'} else machine
    base = f'https://github.com/indygreg/python-build-standalone/releases/download/{PYTHON_STANDALONE_RELEASE}/'
    if os.name == 'nt':
        return base + f'cpython-{PYTHON_STANDALONE_VERSION}+{PYTHON_STANDALONE_RELEASE}-{arch}-pc-windows-msvc-shared-install_only.tar.gz'
    if sys.platform == 'darwin':
        return base + f'cpython-{PYTHON_STANDALONE_VERSION}+{PYTHON_STANDALONE_RELEASE}-{arch}-apple-darwin-install_only.tar.gz'
    if sys.platform.startswith('linux'):
        return base + f'cpython-{PYTHON_STANDALONE_VERSION}+{PYTHON_STANDALONE_RELEASE}-{arch}-unknown-linux-gnu-install_only.tar.gz'
    return None


def _find_python_in_tree(root: Path) -> Path | None:
    if not root.exists():
        return None
    if os.name == 'nt':
        names = {'python.exe'}
    else:
        names = {'python3.11', 'python3'}
    for path in root.rglob('*'):
        if path.is_file() and path.name in names:
            return path
    return None


def _download_file(url: str, destination: Path, heartbeat_cb: Optional[Callable[[], None]] = None) -> None:
    request = urllib.request.Request(url, headers={'User-Agent': 'Modly-UniRig-PythonBootstrap'})
    with urllib.request.urlopen(request) as response, destination.open('wb') as out_file:
        last_heartbeat = 0.0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            out_file.write(chunk)
            now = time.time()
            if heartbeat_cb and now - last_heartbeat >= 2.0:
                heartbeat_cb()
                last_heartbeat = now


def _extract_archive(archive: Path, destination: Path, heartbeat_cb: Optional[Callable[[], None]] = None) -> None:
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                zf.extract(member, destination)
                if heartbeat_cb:
                    heartbeat_cb()
        return
    with tarfile.open(archive, 'r:*') as tf:
        for member in tf.getmembers():
            tf.extract(member, destination)
            if heartbeat_cb:
                heartbeat_cb()


def _marker_path(runtime: RuntimeContext, phase: str) -> Path:
    safe = phase.replace(' ', '_').replace('/', '_')
    return runtime.runtime_root / '.markers' / f'{safe}.done'


def _mark_phase(runtime: RuntimeContext, phase: str) -> None:
    marker = _marker_path(runtime, phase)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(int(time.time())), encoding='utf-8')


def _phase_marked(runtime: RuntimeContext, phase: str) -> bool:
    return _marker_path(runtime, phase).exists()


def _is_venv_valid(runtime: RuntimeContext) -> bool:
    if not runtime.python_exe.exists():
        return False
    try:
        _run_capture([str(runtime.python_exe), '-c', 'import sys; print(sys.executable)'], cwd=runtime.runtime_root)
        return True
    except Exception:
        return False


def _run_subprocess_with_heartbeat(
    runtime: RuntimeContext,
    state: dict,
    progress_cb: Optional[Callable[[int, str], None]],
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    phase: str,
    timeout_sec: int,
    env: dict | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recent_lines: deque[str] = deque(maxlen=100)
    now = int(time.time())
    state.update(
        {
            'phase_started_at': now,
            'last_output_at': now,
            'phase_timeout_sec': timeout_sec,
            'subprocess_cmd': _format_command(command),
            'subprocess_pid': None,
            'subprocess_alive': True,
            'last_error_excerpt': '',
        }
    )
    _write_bootstrap_state(runtime, state)
    _log_bootstrap_event(log_path, phase, 'start', _format_command(command))

    with log_path.open('a', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        state['subprocess_pid'] = process.pid
        _write_bootstrap_state(runtime, state)

        out_queue: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                out_queue.put(line)
            out_queue.put(None)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        last_heartbeat = time.time()
        phase_started = time.time()
        stream_closed = False

        while True:
            try:
                item = out_queue.get(timeout=1.0)
            except queue.Empty:
                item = '__NO_LINE__'

            if item is None:
                stream_closed = True
            elif item != '__NO_LINE__':
                line = item.rstrip('\n')
                recent_lines.append(line)
                log_file.write(item)
                log_file.flush()
                state['last_output_at'] = int(time.time())
                _write_bootstrap_state(runtime, state)

            if process.poll() is None and (time.time() - last_heartbeat) >= HEARTBEAT_INTERVAL_SEC:
                _heartbeat_state(runtime, state, progress_cb)
                _log_bootstrap_event(log_path, phase, 'heartbeat', f'pid={process.pid}')
                last_heartbeat = time.time()

            if process.poll() is None and (time.time() - phase_started) > timeout_sec:
                process.terminate()
                excerpt = '\n'.join(recent_lines) or '(no output captured)'
                state.update({'last_error_excerpt': excerpt, 'subprocess_alive': False})
                _write_bootstrap_state(runtime, state)
                _log_bootstrap_event(log_path, phase, 'error', f'timeout after {timeout_sec}s')
                raise WorkspaceToolError(f'Phase timeout ({phase}, {timeout_sec}s). Last output:\n{excerpt}')

            if process.poll() is not None and stream_closed:
                break

        rc = process.wait()
        if process.stdout:
            process.stdout.close()
        state['subprocess_alive'] = False
        _write_bootstrap_state(runtime, state)
        if rc == 0:
            _log_bootstrap_event(log_path, phase, 'ok')
            return
        excerpt = '\n'.join(recent_lines) or '(no output captured)'
        state['last_error_excerpt'] = excerpt
        _write_bootstrap_state(runtime, state)
        _log_bootstrap_event(log_path, phase, 'error', f'exit={rc}')
        raise WorkspaceToolError(f'Command failed ({phase}, exit={rc}). Last output:\n{excerpt}')


def _bootstrap_runtime(
    runtime: RuntimeContext,
    state: dict,
    bootstrap_log: Path,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    phase_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    _log_bootstrap_event(bootstrap_log, 'bootstrap deps', 'start', 'resumable phases enabled')
    profile, profile_meta = resolve_runtime_profile(runtime)
    state.update(
        {
            'selected_runtime_profile': profile.profile_id,
            'selected_torch_version': profile.torch,
            'selected_cuda_flavor': profile.cuda_flavor,
            'selected_pyg_wheel_url': profile.pyg_wheel_url,
            'selected_torch_index_url': profile.torch_index_url,
            'binary_only_mode': True,
            'source_build_allowed': _env_truthy('MODLY_UNIRIG_ALLOW_SOURCE_BUILDS', False),
        }
    )
    _write_bootstrap_state(runtime, state)
    _log_bootstrap_event(
        bootstrap_log,
        'runtime profile',
        'ok',
        f"{profile_meta.get('reason', 'selected')} (explicitly not using torch 2.11.0+cpu default path)",
    )
    _repair_incompatible_runtime_stack(runtime, state, bootstrap_log, profile, progress_cb=progress_cb)
    if not _phase_marked(runtime, 'upgrade_pip'):
        _run_subprocess_with_heartbeat(
            runtime,
            state,
            progress_cb,
            _pip_install_command(runtime, ['--upgrade', 'pip', 'setuptools', 'wheel']),
            cwd=runtime.repo_dir,
            log_path=bootstrap_log,
            phase='updating pip tooling',
            timeout_sec=PHASE_TIMEOUTS_SEC['installing official requirements'],
        )
        _mark_phase(runtime, 'upgrade_pip')
        _complete_phase(state, 'upgrade_pip')
    if phase_cb:
        phase_cb(68, 'installing torch')
    _install_torch(runtime, state, bootstrap_log, profile, progress_cb=progress_cb)
    if phase_cb:
        phase_cb(76, 'installing official UniRig requirements')
    _install_official_requirements(runtime, state, bootstrap_log, progress_cb=progress_cb)
    if phase_cb:
        phase_cb(84, 'installing spconv/PyG')
    _install_spconv_and_pyg(runtime, state, bootstrap_log, profile, progress_cb=progress_cb)
    _repair_missing_unirig_dependencies(runtime, state, bootstrap_log, progress_cb=progress_cb)
    _log_bootstrap_event(bootstrap_log, 'bootstrap deps', 'ok')


def _is_module_importable(runtime: RuntimeContext, module_name: str) -> bool:
    try:
        _run_capture([str(runtime.python_exe), '-c', f'import {module_name}; print("ok")'], cwd=runtime.repo_dir)
        return True
    except Exception:
        return False


def _repair_incompatible_runtime_stack(
    runtime: RuntimeContext,
    state: dict,
    log_path: Path,
    profile: RuntimeProfile,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    if not runtime.python_exe.exists():
        return
    packages = [
        'torch',
        'torchvision',
        'torchaudio',
        'torch-scatter',
        'torch-cluster',
        'torch-sparse',
        'pyg-lib',
        'spconv',
        'spconv-cu120',
    ]
    try:
        torch_build = _query_torch_build(runtime)
    except Exception:
        torch_build = {}
    try:
        detected_packages = _query_installed_packages(runtime, packages)
    except Exception:
        detected_packages = {}

    installed_torch = str(torch_build.get('version', ''))
    expected_torch = profile.torch
    present_packages = sorted(detected_packages.keys())
    state['repair_detected_packages'] = detected_packages

    incompatible_torch = bool(installed_torch and installed_torch != expected_torch)
    if not present_packages:
        reason = 'no incompatible packages detected'
        state['repair_skipped_reason'] = reason
        state['repair_selected_packages'] = []
        _write_bootstrap_state(runtime, state)
        _log_bootstrap_event(
            log_path,
            'repair runtime stack',
            'ok',
            f"skipped: {reason} (installed={installed_torch or 'missing'}, expected={expected_torch}, detected=none)",
        )
        return

    residual_packages = {
        'torch-scatter',
        'torch-cluster',
        'torch-sparse',
        'pyg-lib',
        'spconv',
        'spconv-cu120',
    }
    residual_detected = sorted(pkg for pkg in present_packages if pkg in residual_packages)
    should_repair = incompatible_torch or (not installed_torch and bool(residual_detected))
    if not should_repair:
        reason = 'compatible stack detected; cleanup not required'
        state['repair_skipped_reason'] = reason
        state['repair_selected_packages'] = []
        _write_bootstrap_state(runtime, state)
        _log_bootstrap_event(
            log_path,
            'repair runtime stack',
            'ok',
            f"skipped: {reason} (installed={installed_torch or 'missing'}, expected={expected_torch}, detected={present_packages})",
        )
        return

    selected_packages = present_packages if incompatible_torch else residual_detected
    if not selected_packages:
        reason = 'no incompatible packages detected'
        state['repair_skipped_reason'] = reason
        state['repair_selected_packages'] = []
        _write_bootstrap_state(runtime, state)
        _log_bootstrap_event(log_path, 'repair runtime stack', 'ok', f'skipped: {reason}')
        return

    state['repair_skipped_reason'] = ''
    state['repair_selected_packages'] = selected_packages
    _write_bootstrap_state(runtime, state)
    _log_bootstrap_event(
        log_path,
        'repair runtime stack',
        'start',
        (
            f"partial uninstall due to incompatible stack (installed={installed_torch or 'missing'}, "
            f"expected={expected_torch}, detected={present_packages}, selected={selected_packages})"
        ),
    )
    _run_subprocess_with_heartbeat(
        runtime,
        state,
        progress_cb,
        _pip_uninstall_command(runtime, ['-y', *selected_packages]),
        cwd=runtime.repo_dir,
        log_path=log_path,
        phase='repairing incompatible torch/PyG stack',
        timeout_sec=PHASE_TIMEOUTS_SEC['installing torch'],
    )
    _log_bootstrap_event(log_path, 'repair runtime stack', 'ok', f'partial cleanup completed ({selected_packages})')


def _install_torch(
    runtime: RuntimeContext,
    state: dict,
    log_path: Path,
    profile: RuntimeProfile,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    if _phase_marked(runtime, 'install_torch') and _is_module_importable(runtime, 'torch'):
        _log_bootstrap_event(log_path, 'installing torch', 'ok', 'skipped: already installed')
        _complete_phase(state, 'install_torch')
        return
    if profile.profile_id.startswith('win-') and '+cpu' in profile.torch:
        raise WorkspaceToolError('Windows UniRig profile cannot use CPU-only torch as default.')
    override_spec = os.environ.get('MODLY_UNIRIG_TORCH_SPEC', '').strip()
    torch_spec = (
        shlex.split(override_spec)
        if override_spec
        else [
            f'torch=={profile.torch}',
            f'torchvision=={profile.torchvision}',
            f'torchaudio=={profile.torchaudio}',
        ]
    )
    cmd = _pip_install_command(runtime, [*torch_spec, '--index-url', profile.torch_index_url])
    _run_subprocess_with_heartbeat(
        runtime,
        state,
        progress_cb,
        cmd,
        cwd=runtime.repo_dir,
        log_path=log_path,
        phase='installing torch',
        timeout_sec=PHASE_TIMEOUTS_SEC['installing torch'],
    )
    _mark_phase(runtime, 'install_torch')
    _complete_phase(state, 'install_torch')


def _install_official_requirements(
    runtime: RuntimeContext,
    state: dict,
    log_path: Path,
    force: bool = False,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    requirements_path = runtime.repo_dir / 'requirements.txt'
    if not requirements_path.exists():
        raise WorkspaceToolError(f'UniRig requirements.txt not found: {requirements_path}')
    state['required_baseline_source'] = str(requirements_path)
    if _phase_marked(runtime, 'official_requirements') and not force:
        _complete_phase(state, 'official_requirements')
        _write_bootstrap_state(runtime, state)
        _log_bootstrap_event(log_path, 'installing official UniRig requirements', 'ok', 'skipped: marker exists')
        return
    install_requirements_path, filtered_flash_attn = _resolve_official_requirements_install_path(runtime, requirements_path)
    if filtered_flash_attn:
        _log_bootstrap_event(
            log_path,
            'installing official UniRig requirements',
            'start',
            'filtered flash_attn requirement (set MODLY_UNIRIG_ENABLE_FLASH_ATTN=1 to keep it)',
        )
    command = _pip_install_command(runtime, ['-r', str(install_requirements_path)])
    try:
        _run_subprocess_with_heartbeat(
            runtime,
            state,
            progress_cb,
            command,
            cwd=runtime.repo_dir,
            log_path=log_path,
            phase='installing official UniRig requirements',
            timeout_sec=PHASE_TIMEOUTS_SEC['installing official requirements'],
        )
    finally:
        if filtered_flash_attn:
            install_requirements_path.unlink(missing_ok=True)
    _mark_phase(runtime, 'official_requirements')
    _complete_phase(state, 'official_requirements')
    state['required_baseline_installed'] = True
    _write_bootstrap_state(runtime, state)


def _resolve_official_requirements_install_path(runtime: RuntimeContext, requirements_path: Path) -> tuple[Path, bool]:
    if _env_truthy('MODLY_UNIRIG_ENABLE_FLASH_ATTN', False):
        return requirements_path, False
    lines = requirements_path.read_text(encoding='utf-8').splitlines()
    filtered_lines: list[str] = []
    filtered = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            filtered_lines.append(line)
            continue
        normalized = stripped.lower()
        if re.match(r'^flash[-_]?attn(\[.*\])?([<>=!~].*)?$', normalized):
            filtered = True
            continue
        filtered_lines.append(line)
    if not filtered:
        return requirements_path, False
    runtime.runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=runtime.runtime_root,
        prefix='requirements_no_flash_attn_',
        suffix='.txt',
        delete=False,
    ) as handle:
        handle.write('\n'.join(filtered_lines) + '\n')
    return Path(handle.name), True


def _install_spconv_and_pyg(
    runtime: RuntimeContext,
    state: dict,
    log_path: Path,
    profile: RuntimeProfile,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    if _phase_marked(runtime, 'spconv_pyg'):
        _complete_phase(state, 'spconv_pyg')
        _log_bootstrap_event(log_path, 'installing spconv/PyG', 'ok', 'skipped: marker exists')
        return
    spconv_package = os.environ.get('MODLY_UNIRIG_SPCONV_PACKAGE', profile.spconv_package)
    _run_subprocess_with_heartbeat(
        runtime,
        state,
        progress_cb,
        _pip_install_command(runtime, [spconv_package, '--only-binary=:all:']),
        cwd=runtime.repo_dir,
        log_path=log_path,
        phase='installing spconv/PyG',
        timeout_sec=PHASE_TIMEOUTS_SEC['installing spconv/PyG'],
    )

    pyg_url = os.environ.get('MODLY_UNIRIG_PYG_WHEEL_URL', profile.pyg_wheel_url)
    binary_only = not _env_truthy('MODLY_UNIRIG_ALLOW_SOURCE_BUILDS', False)
    pyg_packages = [f'torch-scatter=={profile.torch_scatter}', f'torch-cluster=={profile.torch_cluster}']
    install_args = [*pyg_packages, '-f', pyg_url, '--no-cache-dir']
    if binary_only:
        install_args.append('--only-binary=:all:')
    try:
        _run_subprocess_with_heartbeat(
            runtime,
            state,
            progress_cb,
            _pip_install_command(runtime, install_args),
            cwd=runtime.repo_dir,
            log_path=log_path,
            phase='installing spconv/PyG',
            timeout_sec=PHASE_TIMEOUTS_SEC['installing spconv/PyG'],
        )
    except Exception as exc:
        if binary_only:
            raise WorkspaceToolError(
                'Binary PyG wheels were not available for the selected runtime profile. '
                'Source builds are blocked by default. Set MODLY_UNIRIG_ALLOW_SOURCE_BUILDS=1 for expert mode.'
            ) from exc
        raise
    _mark_phase(runtime, 'spconv_pyg')
    _complete_phase(state, 'spconv_pyg')


def _repair_missing_unirig_dependencies(
    runtime: RuntimeContext,
    state: dict,
    log_path: Path,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    missing = _query_missing_required_modules(runtime)
    state['missing_required_modules'] = missing
    _write_bootstrap_state(runtime, state)
    if not missing:
        _log_bootstrap_event(log_path, 'repairing missing UniRig dependencies', 'ok', 'no missing required modules')
        return
    _log_bootstrap_event(
        log_path,
        'repairing missing UniRig dependencies',
        'start',
        f"missing modules detected: {', '.join(missing)}",
    )
    _install_official_requirements(runtime, state, log_path, force=True, progress_cb=progress_cb)
    missing_after = _query_missing_required_modules(runtime)
    state['missing_required_modules'] = missing_after
    _write_bootstrap_state(runtime, state)
    if missing_after:
        raise WorkspaceToolError(
            'Missing required UniRig modules after repair: ' + ', '.join(missing_after)
        )
    _log_bootstrap_event(log_path, 'repairing missing UniRig dependencies', 'ok', 'required modules repaired')


def _pip_install_command(runtime: RuntimeContext, args: Sequence[str]) -> list[str]:
    cmd = [
        str(runtime.python_exe),
        '-m',
        'pip',
        'install',
        '--disable-pip-version-check',
        '--progress-bar',
        'off',
        '--no-input',
        *args,
    ]
    wheelhouse = os.environ.get('MODLY_UNIRIG_WHEELHOUSE_DIR')
    if wheelhouse:
        cmd.extend(['--no-index', '--find-links', wheelhouse])
    return cmd


def _pip_uninstall_command(runtime: RuntimeContext, args: Sequence[str]) -> list[str]:
    return [
        str(runtime.python_exe),
        '-m',
        'pip',
        'uninstall',
        '--disable-pip-version-check',
        '--no-input',
        *args,
    ]


def _query_installed_packages(runtime: RuntimeContext, package_names: Sequence[str]) -> dict[str, str]:
    script = (
        'import json; '
        'import importlib.metadata as md; '
        f'packages={list(package_names)!r}; '
        'found={}; '
        'for name in packages:\n'
        '    try:\n'
        '        found[name]=md.version(name)\n'
        '    except md.PackageNotFoundError:\n'
        '        continue\n'
        'print(json.dumps(found))'
    )
    output = _run_capture([str(runtime.python_exe), '-c', script], cwd=runtime.repo_dir)
    payload = _parse_last_json_line(output, 'Failed to query installed runtime packages')
    return payload if isinstance(payload, dict) else {}


def _query_torch_build(runtime: RuntimeContext) -> dict:
    script = (
        'import json, torch; '
        'version = torch.__version__; '
        "torch_version = version.split('+', 1)[0]; "
        "build = version.split('+', 1)[1] if '+' in version else ''; "
        "cuda_tag = build if build.startswith('cu') else ('cu' + ''.join((torch.version.cuda or '0.0').split('.')[:2]) if torch.version.cuda else 'cpu'); "
        "print(json.dumps({'torch': torch_version, 'version': version, 'cuda': cuda_tag, 'cuda_version': torch.version.cuda}))"
    )
    output = _run_capture([str(runtime.python_exe), '-c', script], cwd=runtime.repo_dir)
    return _parse_last_json_line(output, 'Failed to query torch build information')


def _validate_runtime(runtime: RuntimeContext, bootstrap_log: Path | None = None) -> dict:
    version_script = (
        'import json, sys; '
        "print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor, 'micro': sys.version_info.micro}))"
    )
    version_info = _parse_last_json_line(
        _run_capture([str(runtime.python_exe), '-c', version_script], cwd=runtime.repo_dir),
        'Failed to query the UniRig Python version',
    )
    if (version_info.get('major'), version_info.get('minor')) != (3, 11):
        raise WorkspaceToolError(
            'UniRig requires standalone Python 3.11 inside the isolated runtime. '
            f'Current runtime is {version_info.get("major")}.{version_info.get("minor")}.{version_info.get("micro")}. '
            'Blender Python is not used for this bootstrap. '
            'Update the Python 3.11 path via MODLY_UNIRIG_PYTHON311_BIN or Modly externalPython311Bin.'
        )

    runtime_info = _query_runtime_info(runtime)
    if bootstrap_log:
        _log_bootstrap_event(bootstrap_log, 'validating required imports', 'start')
    import_validation = _query_required_import_validation(runtime)
    if bootstrap_log:
        _log_bootstrap_event(bootstrap_log, 'validating required imports', 'ok', f"validated modules={len(REQUIRED_IMPORTS)}")
        _log_bootstrap_event(bootstrap_log, 'validating run.py entrypoint', 'start')
    runpy_smoke = _validate_runpy_entrypoint(runtime)
    if bootstrap_log:
        _log_bootstrap_event(bootstrap_log, 'validating run.py entrypoint', 'ok')
    if _env_truthy('MODLY_UNIRIG_ENFORCE_GPU', True):
        if not runtime_info.get('cuda'):
            raise WorkspaceToolError(
                'UniRig runtime is installed, but CUDA is not available inside the isolated venv. '
                'Install CUDA-enabled torch wheels (for example via MODLY_UNIRIG_TORCH_INDEX_URL) or disable enforcement with MODLY_UNIRIG_ENFORCE_GPU=0.'
            )
        min_vram = _safe_float(os.environ.get('MODLY_UNIRIG_MIN_VRAM_GB'), DEFAULT_MIN_VRAM_GB)
        vram_gb = _safe_float(runtime_info.get('vram_gb'), 0.0)
        if vram_gb < min_vram:
            raise WorkspaceToolError(
                f'UniRig requires at least {min_vram:.1f} GB of VRAM. Detected: {vram_gb:.2f} GB.'
            )

    required_paths = [
        runtime.repo_dir / 'run.py',
        runtime.repo_dir / SKELETON_TASK,
        runtime.repo_dir / SKIN_TASK,
        runtime.repo_dir / 'src' / 'inference' / 'merge.py',
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise WorkspaceToolError(f'UniRig runtime is incomplete. Missing files: {missing}')
    return {
        'python': version_info,
        'gpu': runtime_info,
        'imports': import_validation,
        'runpy_smoke_ok': runpy_smoke['ok'],
        'missing_required_modules': import_validation.get('missing_modules', []),
        'baseline_installed': not bool(import_validation.get('missing_modules')),
    }


def _query_runtime_info(runtime: RuntimeContext) -> dict:
    script = (
        'import json, torch; '
        "payload = {'cuda': bool(torch.cuda.is_available())}; "
        "payload['torch_version'] = torch.__version__; "
        "payload['cuda_version'] = torch.version.cuda; "
        '\nif torch.cuda.is_available():\n'
        '    props = torch.cuda.get_device_properties(0)\n'
        "    payload['gpu_name'] = props.name\n"
        "    payload['vram_gb'] = round(props.total_memory / (1024 ** 3), 3)\n"
        'print(json.dumps(payload))'
    )
    output = _run_capture([str(runtime.python_exe), '-c', script], cwd=runtime.repo_dir)
    return _parse_last_json_line(output, 'Failed to query UniRig GPU runtime information')


def _query_missing_required_modules(runtime: RuntimeContext) -> list[str]:
    validation = _query_required_import_validation(runtime, raise_on_missing=False)
    missing = validation.get('missing_modules', [])
    return missing if isinstance(missing, list) else []


def _query_required_import_validation(runtime: RuntimeContext, raise_on_missing: bool = True) -> dict:
    requirements = [{'module': module_name, 'statement': statement} for module_name, statement in REQUIRED_IMPORTS]
    script = (
        'import importlib, json; '
        f'requirements={requirements!r}; '
        'results = {}; '
        'missing = []; '
        '\nfor item in requirements:\n'
        "    mod = item['module']\n"
        '    try:\n'
        '        importlib.import_module(mod)\n'
        "        results[mod] = 'ok'\n"
        '    except Exception as exc:\n'
        "        results[mod] = f'error: {exc}'\n"
        '        missing.append(mod)\n'
        "payload = {'results': results, 'missing_modules': missing}; "
        'print(json.dumps(payload))'
    )
    output = _run_capture([str(runtime.python_exe), '-c', script], cwd=runtime.repo_dir)
    payload = _parse_last_json_line(output, 'Failed to validate required UniRig imports')
    missing = payload.get('missing_modules', [])
    if raise_on_missing and missing:
        raise WorkspaceToolError(f"UniRig runtime import validation failed. Missing modules: {', '.join(missing)}")
    return payload


def _validate_runpy_entrypoint(runtime: RuntimeContext) -> dict:
    command = [str(runtime.python_exe), 'run.py', '--help']
    try:
        output = _run_capture(command, cwd=runtime.repo_dir)
        return {'ok': True, 'output_excerpt': '\n'.join(output.splitlines()[:20])}
    except Exception as exc:
        raise WorkspaceToolError(f'run.py entrypoint validation failed: {exc}') from exc


def _prepare_input_mesh(input_path: Path, temp_root: Path) -> Path:
    suffix = input_path.suffix.lower()
    if suffix in DIRECT_INPUT_SUFFIXES:
        return input_path.resolve()
    if suffix not in CONVERTIBLE_INPUT_SUFFIXES:
        raise WorkspaceToolError(f'Unsupported UniRig input format: {suffix}')

    converted_path = temp_root / f'{input_path.stem}_prepared.glb'
    try:
        force = 'scene' if suffix == '.gltf' else None
        loaded = trimesh.load(input_path, force=force)
        if isinstance(loaded, trimesh.Trimesh):
            scene = loaded.scene()
        elif isinstance(loaded, trimesh.Scene):
            scene = loaded
        else:
            raise WorkspaceToolError(f'Unsupported trimesh payload while converting {input_path.name}: {type(loaded)!r}')
        blob = scene.export(file_type='glb')
        converted_path.write_bytes(blob if isinstance(blob, (bytes, bytearray)) else bytes(blob))
    except WorkspaceToolError:
        raise
    except Exception as exc:
        raise WorkspaceToolError(f'Failed to convert {input_path.name} to GLB for UniRig: {exc}') from exc
    return converted_path


def _run_logged(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    env: dict | None = None,
    phase: str = 'command',
    heartbeat_cb: Optional[Callable[[], None]] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log_file:
        log_file.write(f'[{phase}] start\n')
        log_file.write(f'$ {_format_command(command)}\n')
        log_file.flush()
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if heartbeat_cb:
            heartbeat_cb()
        if result.returncode == 0:
            log_file.write(f'[{phase}] ok\n')
        else:
            log_file.write(f'[{phase}] error (exit={result.returncode})\n')
    if result.returncode != 0:
        tail = _tail_text(log_path, 60)
        raise WorkspaceToolError(
            f'Command failed while running UniRig: {_format_command(command)}\n\nLast output:\n{tail}'
        )


def _run_capture(command: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = '\n'.join((result.stderr or result.stdout).splitlines()[-60:])
        raise WorkspaceToolError(f'Command failed: {_format_command(command)}\n\nLast output:\n{tail}')
    return result.stdout or result.stderr or ''


def _parse_last_json_line(output: str, context: str) -> dict:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise WorkspaceToolError(f'{context}. Raw output:\n{output[-2000:]}')


def _tail_text(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(content[-lines:])
    except Exception:
        return '(unable to read log file)'


def _format_command(command: Sequence[str]) -> str:
    return ' '.join(_quote_arg(arg) for arg in command)


def _quote_arg(value: object) -> str:
    text = str(value)
    if os.name == 'nt':
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {'0', 'false', 'no', 'off', ''}


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _log_bootstrap_event(log_path: Path, phase: str, status: str, detail: str = '') -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    line = f'[{timestamp}] [{phase}] {status}'
    if detail:
        line += f' :: {detail}'
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


__all__ = [
    'UniRigWorkspaceTool',
    'UniRigStageCommands',
    'build_unirig_commands',
    '_prepare_input_mesh',
    '_default_bootstrap_state',
    '_normalize_state',
    '_resolve_python311_command',
]
