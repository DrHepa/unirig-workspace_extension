from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import trimesh

from services.workspace_tools_base import BaseWorkspaceTool, WorkspaceToolError


OFFICIAL_UNIRIG_ARCHIVE_URL = 'https://api.github.com/repos/VAST-AI-Research/UniRig/tarball/HEAD'
BOOTSTRAP_VERSION = 2
DEFAULT_TORCH_SPEC = 'torch torchvision torchaudio'
DEFAULT_SPCONV_PACKAGE = 'spconv-cu120'
DEFAULT_PYG_PACKAGES = ('torch-scatter', 'torch-cluster')
DEFAULT_MIN_VRAM_GB = 8.0
INSTALL_STATES = {'not_installed', 'installing', 'ready', 'error'}

DIRECT_INPUT_SUFFIXES = {'.obj', '.fbx', '.glb', '.vrm'}
CONVERTIBLE_INPUT_SUFFIXES = {'.gltf', '.stl', '.ply'}
SUPPORTED_SUFFIXES = DIRECT_INPUT_SUFFIXES | CONVERTIBLE_INPUT_SUFFIXES

SKELETON_TASK = 'configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml'
SKIN_TASK = 'configs/task/quick_inference_unirig_skin.yaml'
SKIN_DATA_NAME = 'raw_data.npz'
MERGE_REQUIRE_SUFFIX = 'obj,fbx,FBX,dae,glb,gltf,vrm'


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

        try:
            self._report(progress_cb, 5, 'Preparing runtime directories…')
            _set_state(runtime, state, step='resolve_python', percent=10)
            python_cmd = _resolve_python311_command()

            _set_state(runtime, state, step='prepare_repo', percent=20)
            self._prepare_repo(runtime)

            _set_state(runtime, state, step='create_venv', percent=35)
            _create_venv(runtime, python_cmd)

            _set_state(runtime, state, step='bootstrap_deps', percent=60)
            _bootstrap_runtime(runtime, bootstrap_log)

            _set_state(runtime, state, step='validate', percent=85)
            validation = _validate_runtime(runtime)

            _set_state(
                runtime,
                state,
                install_state='ready',
                step='done',
                percent=100,
                completed_at=int(time.time()),
                last_error='',
                python_exe=str(runtime.python_exe),
            )
            state['validation'] = validation
            _write_bootstrap_state(runtime, state)
            self._runtime_ctx = runtime
            self._report(progress_cb, 100, 'Runtime ready')
            return state
        except Exception as exc:
            _set_state(
                runtime,
                state,
                install_state='error',
                step='failed',
                completed_at=int(time.time()),
                last_error=str(exc),
            )
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

    def _prepare_repo(self, runtime: RuntimeContext) -> None:
        if runtime.repo_dir.exists() and (runtime.repo_dir / 'run.py').exists():
            return
        if runtime.external_repo:
            raise WorkspaceToolError(f'UniRig repo override is invalid: {runtime.repo_dir}')
        _download_unirig_repo(runtime)

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


def _bootstrap_state_path(runtime: RuntimeContext) -> Path:
    return runtime.runtime_root / 'bootstrap_state.json'


def _default_bootstrap_state(runtime: RuntimeContext) -> dict:
    return {
        'bootstrap_version': BOOTSTRAP_VERSION,
        'install_state': 'not_installed',
        'percent': 0,
        'step': 'idle',
        'last_error': '',
        'started_at': None,
        'completed_at': None,
        'bootstrap_log': '',
        'python_exe': str(runtime.python_exe),
        'repo_dir': str(runtime.repo_dir),
        'venv_dir': str(runtime.venv_dir),
        'runtime_root': str(runtime.runtime_root),
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


def _download_unirig_repo(runtime: RuntimeContext) -> None:
    archive_url = os.environ.get('MODLY_UNIRIG_REPO_ARCHIVE_URL', OFFICIAL_UNIRIG_ARCHIVE_URL)
    tmp_dir = runtime.runtime_root / '_repo_download'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_dir / 'unirig.tar.gz'

    request = urllib.request.Request(
        archive_url,
        headers={
            'User-Agent': 'Modly-UniRig-WorkspaceTool',
            'Accept': 'application/vnd.github+json',
        },
    )
    try:
        with urllib.request.urlopen(request) as response, archive_path.open('wb') as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as exc:
        raise WorkspaceToolError(
            'Failed to download the official UniRig repository. '
            'Set MODLY_UNIRIG_REPO_DIR to an already downloaded local checkout if you want a fully offline bootstrap. '
            f'Original error: {exc}'
        ) from exc

    extract_dir = tmp_dir / 'extract'
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, 'r:gz') as tar_file:
            tar_file.extractall(extract_dir)
    except Exception as exc:
        raise WorkspaceToolError(f'Failed to extract UniRig repository archive: {exc}') from exc

    candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
    if len(candidates) != 1:
        raise WorkspaceToolError('Unexpected UniRig archive layout while extracting repository.')

    if runtime.repo_dir.exists():
        shutil.rmtree(runtime.repo_dir, ignore_errors=True)
    shutil.copytree(candidates[0], runtime.repo_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_python311_command() -> list[str]:
    settings = _read_modly_settings()

    env_python = os.environ.get('MODLY_UNIRIG_PYTHON311_BIN')
    if env_python:
        cmd = _split_command(env_python)
        _assert_python311(cmd, source='MODLY_UNIRIG_PYTHON311_BIN')
        return cmd

    for key in ('externalPython311Bin', 'python311Bin', 'external_python_311_bin'):
        candidate = settings.get(key)
        if isinstance(candidate, str) and candidate.strip():
            cmd = _split_command(candidate)
            _assert_python311(cmd, source=f'settings.{key}')
            return cmd

    if os.name == 'nt' and shutil.which('py'):
        cmd = ['py', '-3.11']
        if _is_python311(cmd):
            return cmd

    for candidate in ('python3.11', 'python'):
        resolved = shutil.which(candidate)
        if not resolved:
            continue
        cmd = [resolved]
        if _is_python311(cmd):
            return cmd

    raise WorkspaceToolError(
        'Python 3.11 not found. Set MODLY_UNIRIG_PYTHON311_BIN or configure Modly setting '
        'externalPython311Bin to a valid Python 3.11 executable.'
    )


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise WorkspaceToolError('Python 3.11 command is empty.')
    return parts


def _assert_python311(command: list[str], source: str) -> None:
    if not _is_python311(command):
        raise WorkspaceToolError(f'{source} does not point to a Python 3.11 executable: {_format_command(command)}')


def _is_python311(command: Sequence[str]) -> bool:
    try:
        result = subprocess.run(
            [*command, '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return (result.stdout or '').strip() == '3.11'


def _create_venv(runtime: RuntimeContext, python311_command: list[str]) -> None:
    if runtime.venv_dir.exists():
        shutil.rmtree(runtime.venv_dir, ignore_errors=True)
    runtime.venv_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [*python311_command, '-m', 'venv', str(runtime.venv_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = '\n'.join((result.stderr or result.stdout).splitlines()[-40:])
        raise WorkspaceToolError(f'Failed to create the isolated UniRig venv. Last output:\n{tail}')
    if not runtime.python_exe.exists():
        raise WorkspaceToolError(f'Venv creation succeeded but Python executable was not found: {runtime.python_exe}')


def _bootstrap_runtime(runtime: RuntimeContext, bootstrap_log: Path) -> None:
    _run_logged([str(runtime.python_exe), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], cwd=runtime.repo_dir, log_path=bootstrap_log)
    _install_torch(runtime, bootstrap_log)
    _install_filtered_requirements(runtime, bootstrap_log)
    _run_logged([str(runtime.python_exe), '-m', 'pip', 'install', 'numpy==1.26.4', 'scipy'], cwd=runtime.repo_dir, log_path=bootstrap_log)
    _install_spconv_and_pyg(runtime, bootstrap_log)


def _install_torch(runtime: RuntimeContext, log_path: Path) -> None:
    torch_spec = os.environ.get('MODLY_UNIRIG_TORCH_SPEC', DEFAULT_TORCH_SPEC)
    cmd = [str(runtime.python_exe), '-m', 'pip', 'install', *shlex.split(torch_spec)]
    torch_index_url = os.environ.get('MODLY_UNIRIG_TORCH_INDEX_URL')
    if torch_index_url:
        cmd.extend(['--index-url', torch_index_url])
    _run_logged(cmd, cwd=runtime.repo_dir, log_path=log_path)


def _install_filtered_requirements(runtime: RuntimeContext, log_path: Path) -> None:
    requirements_path = runtime.repo_dir / 'requirements.txt'
    if not requirements_path.exists():
        raise WorkspaceToolError(f'UniRig requirements.txt not found: {requirements_path}')

    filtered_text = _filtered_requirements_text(requirements_path.read_text(encoding='utf-8'))
    filtered_path = runtime.runtime_root / 'requirements.filtered.txt'
    filtered_path.write_text(filtered_text, encoding='utf-8')
    _run_logged([str(runtime.python_exe), '-m', 'pip', 'install', '-r', str(filtered_path)], cwd=runtime.repo_dir, log_path=log_path)


def _filtered_requirements_text(raw_text: str) -> str:
    keep_flash_attn = _env_truthy('MODLY_UNIRIG_ENABLE_FLASH_ATTN', False)
    out_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            out_lines.append(line)
            continue
        package_name = _base_requirement_name(stripped)
        if not keep_flash_attn and package_name in {'flash_attn', 'flash-attn'}:
            continue
        out_lines.append(line)
    return '\n'.join(out_lines).rstrip() + '\n'


def _base_requirement_name(requirement_line: str) -> str:
    cleaned = requirement_line.split(';', 1)[0].strip()
    for token in ('==', '>=', '<=', '~=', '!=', '>', '<'):
        if token in cleaned:
            cleaned = cleaned.split(token, 1)[0].strip()
            break
    if '[' in cleaned:
        cleaned = cleaned.split('[', 1)[0].strip()
    return cleaned.replace('_', '-').lower()


def _install_spconv_and_pyg(runtime: RuntimeContext, log_path: Path) -> None:
    spconv_package = os.environ.get('MODLY_UNIRIG_SPCONV_PACKAGE', DEFAULT_SPCONV_PACKAGE)
    _run_logged([str(runtime.python_exe), '-m', 'pip', 'install', spconv_package], cwd=runtime.repo_dir, log_path=log_path)

    pyg_url = os.environ.get('MODLY_UNIRIG_PYG_WHEEL_URL')
    if not pyg_url:
        torch_build = _query_torch_build(runtime)
        pyg_url = f"https://data.pyg.org/whl/torch-{torch_build['torch']}+{torch_build['cuda']}.html"
    cmd = [
        str(runtime.python_exe),
        '-m',
        'pip',
        'install',
        *DEFAULT_PYG_PACKAGES,
        '-f',
        pyg_url,
        '--no-cache-dir',
    ]
    _run_logged(cmd, cwd=runtime.repo_dir, log_path=log_path)


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


def _validate_runtime(runtime: RuntimeContext) -> dict:
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
            'UniRig expects Python 3.11 inside the isolated runtime. '
            f'Current runtime is {version_info.get("major")}.{version_info.get("minor")}.{version_info.get("micro")}. '
            'Update the Python 3.11 path via MODLY_UNIRIG_PYTHON311_BIN or Modly externalPython311Bin.'
        )

    runtime_info = _query_runtime_info(runtime)
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
    return {'python': version_info, 'gpu': runtime_info}


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


def _run_logged(command: Sequence[str], cwd: Path, log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log_file:
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


__all__ = [
    'UniRigWorkspaceTool',
    'UniRigStageCommands',
    'build_unirig_commands',
    '_filtered_requirements_text',
    '_prepare_input_mesh',
    '_default_bootstrap_state',
    '_normalize_state',
    '_resolve_python311_command',
]
