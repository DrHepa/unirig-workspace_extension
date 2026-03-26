import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


class _BaseWorkspaceTool:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir

    def unload(self) -> None:
        return None


class _WorkspaceToolError(Exception):
    pass


services_mod = types.ModuleType('services')
workspace_mod = types.ModuleType('services.workspace_tools_base')
workspace_mod.BaseWorkspaceTool = _BaseWorkspaceTool
workspace_mod.WorkspaceToolError = _WorkspaceToolError
sys.modules.setdefault('services', services_mod)
sys.modules['services.workspace_tools_base'] = workspace_mod


trimesh_mod = types.ModuleType('trimesh')

class _FakeTrimesh:
    def scene(self):
        return self

    def export(self, file_type='glb'):
        return b''


class _FakeScene:
    def export(self, file_type='glb'):
        return b''


def _fake_load(*args, **kwargs):
    return _FakeScene()


trimesh_mod.Trimesh = _FakeTrimesh
trimesh_mod.Scene = _FakeScene
trimesh_mod.load = _fake_load
sys.modules['trimesh'] = trimesh_mod

generator = importlib.import_module('generator')


class RuntimeLifecycleTests(unittest.TestCase):
    def test_runtime_profile_resolver_prefers_win_cu128_on_windows_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                with patch('generator.platform.system', return_value='Windows'), patch(
                    'generator._detect_cuda_environment',
                    return_value={'has_nvidia_gpu': True, 'nvidia_smi_ok': True},
                ), patch('generator._validate_profile_binary_wheels', side_effect=[{'ok': True, 'reason': 'ok'}]):
                    profile, _meta = generator.resolve_runtime_profile(runtime)
                self.assertEqual(profile.profile_id, 'win-cu128-stable')

    def test_runtime_profile_rejects_cpu_default_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                with patch('generator.platform.system', return_value='Windows'), patch(
                    'generator._detect_cuda_environment',
                    return_value={'has_nvidia_gpu': False, 'nvidia_smi_ok': False},
                ):
                    with self.assertRaises(_WorkspaceToolError) as ctx:
                        generator.resolve_runtime_profile(runtime)
                self.assertIn('CPU torch profile is not used by default', str(ctx.exception))

    def test_spconv_pyg_binary_only_failure_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                state = generator._default_bootstrap_state(runtime)
                profile = generator.get_supported_runtime_profiles()[0]
                with patch('generator._run_subprocess_with_heartbeat', side_effect=[None, _WorkspaceToolError('no wheel')]):
                    with self.assertRaises(_WorkspaceToolError) as ctx:
                        generator._install_spconv_and_pyg(runtime, state, runtime.runtime_root / 'bootstrap.log', profile)
                self.assertIn('Source builds are blocked by default', str(ctx.exception))

    def test_repair_runtime_stack_uninstalls_incompatible_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                state = generator._default_bootstrap_state(runtime)
                profile = generator.get_supported_runtime_profiles()[0]
                called: list[list[str]] = []

                def capture_cmd(*args: Any, **kwargs: Any) -> None:
                    called.append(list(args[3]))

                with patch('generator._query_torch_build', return_value={'version': '2.11.0+cpu'}), patch(
                    'generator._query_installed_packages',
                    return_value={'torch': '2.11.0+cpu', 'torch-scatter': '2.1.2', 'spconv-cu120': '2.3.8'},
                ), patch(
                    'generator._run_subprocess_with_heartbeat',
                    side_effect=capture_cmd,
                ):
                    generator._repair_incompatible_runtime_stack(runtime, state, runtime.runtime_root / 'bootstrap.log', profile)
                self.assertTrue(any('uninstall' in cmd for cmd in called))
                uninstall_cmd = next(cmd for cmd in called if 'uninstall' in cmd)
                self.assertIn('torch', uninstall_cmd)
                self.assertIn('torch-scatter', uninstall_cmd)
                self.assertIn('spconv-cu120', uninstall_cmd)
                self.assertEqual(state['repair_selected_packages'], ['spconv-cu120', 'torch', 'torch-scatter'])

    def test_pip_uninstall_command_excludes_progress_bar_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                cmd = generator._pip_uninstall_command(runtime, ['-y', 'torch'])
                self.assertIn('uninstall', cmd)
                self.assertNotIn('--progress-bar', cmd)
                self.assertIn('--no-input', cmd)

    def test_pip_install_command_includes_progress_bar_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                cmd = generator._pip_install_command(runtime, ['torch'])
                self.assertIn('install', cmd)
                self.assertIn('--progress-bar', cmd)
                self.assertIn('off', cmd)

    def test_resolve_official_requirements_filters_flash_attn_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                requirements = runtime.repo_dir / 'requirements.txt'
                requirements.write_text('numpy==1.26.4\nflash_attn==2.6.3\ntorch==2.6.0\n', encoding='utf-8')
                install_path, filtered = generator._resolve_official_requirements_install_path(runtime, requirements)
                self.assertTrue(filtered)
                self.assertNotEqual(install_path, requirements)
                self.assertTrue(install_path.exists())
                content = install_path.read_text(encoding='utf-8')
                self.assertIn('numpy==1.26.4', content)
                self.assertIn('torch==2.6.0', content)
                self.assertNotIn('flash_attn==2.6.3', content)

    def test_resolve_official_requirements_keeps_flash_attn_with_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {'MODLY_UNIRIG_RUNTIME_DIR': tmp, 'MODLY_UNIRIG_ENABLE_FLASH_ATTN': '1'},
                clear=False,
            ):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                requirements = runtime.repo_dir / 'requirements.txt'
                requirements.write_text('numpy==1.26.4\nflash-attn==2.6.3\n', encoding='utf-8')
                install_path, filtered = generator._resolve_official_requirements_install_path(runtime, requirements)
                self.assertFalse(filtered)
                self.assertEqual(install_path, requirements)

    def test_repair_runtime_stack_skips_uninstall_when_no_packages_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                state = generator._default_bootstrap_state(runtime)
                profile = generator.get_supported_runtime_profiles()[0]
                with patch('generator._query_torch_build', return_value={}), patch(
                    'generator._query_installed_packages',
                    return_value={},
                ), patch('generator._run_subprocess_with_heartbeat') as run_mock:
                    generator._repair_incompatible_runtime_stack(runtime, state, runtime.runtime_root / 'bootstrap.log', profile)
                run_mock.assert_not_called()
                self.assertEqual(state['repair_selected_packages'], [])
                self.assertEqual(state['repair_skipped_reason'], 'no incompatible packages detected')
                self.assertEqual(state['repair_detected_packages'], {})

    def test_repair_runtime_stack_selective_uninstall_for_residual_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                state = generator._default_bootstrap_state(runtime)
                profile = generator.get_supported_runtime_profiles()[0]
                called: list[list[str]] = []

                def capture_cmd(*args: Any, **kwargs: Any) -> None:
                    called.append(list(args[3]))

                detected = {'torch-scatter': '2.1.2', 'spconv-cu120': '2.3.8'}
                with patch('generator._query_torch_build', return_value={}), patch(
                    'generator._query_installed_packages',
                    return_value=detected,
                ), patch('generator._run_subprocess_with_heartbeat', side_effect=capture_cmd):
                    generator._repair_incompatible_runtime_stack(runtime, state, runtime.runtime_root / 'bootstrap.log', profile)
                uninstall_cmd = next(cmd for cmd in called if 'uninstall' in cmd)
                self.assertIn('torch-scatter', uninstall_cmd)
                self.assertIn('spconv-cu120', uninstall_cmd)
                self.assertNotIn('torch', uninstall_cmd)
                self.assertEqual(state['repair_selected_packages'], ['spconv-cu120', 'torch-scatter'])

    def test_repair_runtime_stack_log_and_state_reflect_skip_vs_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                profile = generator.get_supported_runtime_profiles()[0]
                log_path = runtime.runtime_root / 'bootstrap.log'

                skip_state = generator._default_bootstrap_state(runtime)
                with patch('generator._query_torch_build', return_value={}), patch(
                    'generator._query_installed_packages',
                    return_value={},
                ):
                    generator._repair_incompatible_runtime_stack(runtime, skip_state, log_path, profile)
                self.assertEqual(skip_state['repair_skipped_reason'], 'no incompatible packages detected')

                repair_state = generator._default_bootstrap_state(runtime)
                with patch('generator._query_torch_build', return_value={}), patch(
                    'generator._query_installed_packages',
                    return_value={'torch-scatter': '2.1.2'},
                ), patch('generator._run_subprocess_with_heartbeat', return_value=None):
                    generator._repair_incompatible_runtime_stack(runtime, repair_state, log_path, profile)
                self.assertEqual(repair_state['repair_selected_packages'], ['torch-scatter'])
                text = log_path.read_text(encoding='utf-8')
                self.assertIn('skipped: no incompatible packages detected', text)
                self.assertIn('selected=[\'torch-scatter\']', text)

    def test_validate_runtime_import_validation_is_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'run.py').write_text('# stub', encoding='utf-8')
                (runtime.repo_dir / 'configs' / 'task').mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / generator.SKELETON_TASK).write_text('', encoding='utf-8')
                (runtime.repo_dir / generator.SKIN_TASK).write_text('', encoding='utf-8')
                (runtime.repo_dir / 'src' / 'inference').mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'src' / 'inference' / 'merge.py').write_text('', encoding='utf-8')
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                with patch('generator._run_capture', return_value='{"major":3,"minor":11,"micro":9}\n'), patch(
                    'generator._query_runtime_info',
                    return_value={'cuda': True, 'vram_gb': 16.0},
                ), patch(
                    'generator._query_required_import_validation',
                    return_value={'results': {'torch': 'ok'}, 'missing_modules': []},
                ), patch(
                    'generator._validate_runpy_entrypoint',
                    return_value={'ok': True},
                ):
                    payload = generator._validate_runtime(runtime)
                self.assertIn('imports', payload)
                self.assertTrue(payload['runpy_smoke_ok'])

    def test_set_state_and_report_updates_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                state = generator._default_bootstrap_state(runtime)
                seen: list[tuple[int, str]] = []
                generator._set_state_and_report(
                    runtime,
                    state,
                    lambda pct, step: seen.append((pct, step)),
                    33,
                    'finding python 3.11',
                )
                loaded = generator._load_bootstrap_state(runtime)
                self.assertEqual(loaded['percent'], 33)
                self.assertEqual(loaded['step'], 'finding python 3.11')
                self.assertIn('updated_at', loaded)
                self.assertIn('last_heartbeat_at', loaded)
                self.assertEqual(seen, [(33, 'finding python 3.11')])

    def test_runtime_status_without_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                status = tool.runtime_status()
                self.assertEqual(status['install_state'], 'not_installed')
                self.assertEqual(status['percent'], 0)
                self.assertEqual(status['step'], 'idle')
                self.assertIn('message', status)
                self.assertIn('updated_at', status)
                self.assertIn('last_heartbeat_at', status)
                self.assertIn('bootstrap_log', status)
                self.assertIn('selected_python_source', status)
                self.assertIn('selected_python', status)
                self.assertIn('selected_python_version', status)
                self.assertTrue(status['runtime_root'].endswith(tmp))

    def test_install_runtime_errors_when_python311_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    'MODLY_UNIRIG_RUNTIME_DIR': tmp,
                    'MODLY_UNIRIG_PYTHON311_BIN': '/definitely/missing/python311',
                },
                clear=False,
            ):
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                with patch('generator._ensure_runtime_python311', return_value=None), \
                     patch('generator._find_modly_bundled_python', return_value=None), \
                     patch('generator.shutil.which', return_value=None):
                    with self.assertRaises(_WorkspaceToolError):
                        tool.install_runtime()
                state_path = Path(tmp) / 'bootstrap_state.json'
                state = json.loads(state_path.read_text(encoding='utf-8'))
                self.assertEqual(state['install_state'], 'error')
                self.assertIn('standalone Python 3.11', state['last_error'])

    def test_bootstrap_state_persistence_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                payload = generator._default_bootstrap_state(runtime)
                payload['install_state'] = 'installing'
                payload['percent'] = 42
                generator._write_bootstrap_state(runtime, payload)

                loaded = generator._load_bootstrap_state(runtime)
                self.assertEqual(loaded['install_state'], 'installing')
                self.assertEqual(loaded['percent'], 42)

    def test_detects_modly_bundled_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundled = root / 'resources' / 'python-embed' / 'bin' / 'python3.11'
            bundled.parent.mkdir(parents=True, exist_ok=True)
            bundled.write_text('', encoding='utf-8')
            with patch('generator._is_python311', return_value=True):
                with patch.dict(os.environ, {'MODLY_RESOURCES_DIR': str(root)}, clear=False):
                    found = generator._find_modly_bundled_python()
            self.assertEqual(found, bundled)

    def test_runtime_local_python_fallback_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()

                def fake_probe(command: list[str]) -> dict[str, Any]:
                    return {'ok': True, 'version': '3.11.9', 'major': 3, 'minor': 11, 'micro': 9}

                with patch('generator._find_modly_bundled_python', return_value=None), \
                     patch('generator._ensure_runtime_python311', return_value=runtime.runtime_root / 'python311' / 'bin' / 'python3.11'), \
                     patch('generator.shutil.which', return_value=None), \
                     patch('generator._probe_python_version', side_effect=fake_probe):
                    payload = generator._resolve_python311_command(runtime)
                self.assertEqual(payload['selected_python_source'], 'runtime_local_python')

    def test_rejects_python314_with_blender_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp, 'MODLY_UNIRIG_PYTHON311_BIN': '/usr/bin/python'}, clear=False):
                runtime = generator._resolve_runtime_context()

                def fake_probe(command: list[str]) -> dict[str, Any]:
                    executable = command[0]
                    if executable == '/usr/bin/python':
                        return {'ok': True, 'version': '3.14.0', 'major': 3, 'minor': 14, 'micro': 0}
                    return {'ok': False, 'reason': 'missing'}

                with patch('generator._find_modly_bundled_python', return_value=None), \
                     patch('generator._ensure_runtime_python311', return_value=None), \
                     patch('generator.shutil.which', return_value=None), \
                     patch('generator._probe_python_version', side_effect=fake_probe):
                    with self.assertRaises(_WorkspaceToolError) as ctx:
                        generator._resolve_python311_command(runtime)
            self.assertIn('Blender Python is not used for this bootstrap', str(ctx.exception))
            self.assertIn('Rejected version 3.14.0', str(ctx.exception))

    def test_bootstrap_state_has_selected_python_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                payload = generator._default_bootstrap_state(runtime)
                payload['selected_python_source'] = 'runtime_local_python'
                generator._write_bootstrap_state(runtime, payload)
                loaded = generator._load_bootstrap_state(runtime)
                self.assertEqual(loaded['selected_python_source'], 'runtime_local_python')

    def test_runtime_status_returns_persisted_intermediate_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                payload = generator._default_bootstrap_state(runtime)
                payload.update(
                    {
                        'install_state': 'installing',
                        'percent': 68,
                        'step': 'installing torch',
                        'message': 'installing torch',
                        'bootstrap_log': str(runtime.logs_dir / 'bootstrap.log'),
                    }
                )
                generator._write_bootstrap_state(runtime, payload)
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                status = tool.runtime_status()
                self.assertEqual(status['install_state'], 'installing')
                self.assertEqual(status['percent'], 68)
                self.assertEqual(status['step'], 'installing torch')

    def test_install_runtime_persists_ready_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'run.py').write_text('# stub', encoding='utf-8')
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                progress: list[tuple[int, str]] = []
                with patch('generator._resolve_python311_command', return_value={
                    'command': ['python3.11'],
                    'selected_python': 'python3.11',
                    'selected_python_version': '3.11.9',
                    'selected_python_source': 'PATH_python',
                    'attempts': [],
                    'phases': [],
                }), patch('generator._run_subprocess_with_heartbeat', return_value=None), patch('generator._bootstrap_runtime', return_value=None), patch(
                    'generator._validate_runtime', return_value={'python': {'major': 3, 'minor': 11}, 'gpu': {'cuda': True}}
                ):
                    status = tool.install_runtime(progress_cb=lambda pct, step: progress.append((pct, step)))
                self.assertEqual(status['install_state'], 'ready')
                saved = generator._load_bootstrap_state(runtime)
                self.assertEqual(saved['install_state'], 'ready')
                self.assertEqual(saved['step'], 'ready')
                self.assertGreaterEqual(saved['percent'], 100)
                self.assertTrue(progress)

    def test_download_repo_phase_is_reported_before_network_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                observed_steps: list[str] = []

                def phase_cb(_percent: int, step: str) -> None:
                    observed_steps.append(step)

                def fake_download(_url: str, _destination: Path, heartbeat_cb=None) -> None:
                    self.assertEqual(observed_steps, ['downloading UniRig repo'])
                    if heartbeat_cb:
                        heartbeat_cb()

                def fake_extract(_archive: Path, destination: Path, heartbeat_cb=None) -> None:
                    extracted_repo = destination / 'unirig-main'
                    extracted_repo.mkdir(parents=True, exist_ok=True)
                    (extracted_repo / 'run.py').write_text('# stub', encoding='utf-8')
                    if heartbeat_cb:
                        heartbeat_cb()

                with patch('generator._download_file', side_effect=fake_download), patch(
                    'generator._extract_archive',
                    side_effect=fake_extract,
                ):
                    generator._download_unirig_repo(runtime, phase_cb=phase_cb)

                self.assertEqual(observed_steps[:2], ['downloading UniRig repo', 'extracting UniRig repo'])
                self.assertTrue((runtime.repo_dir / 'run.py').exists())

    def test_download_repo_forwards_heartbeat_to_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                heartbeat_calls = {'count': 0}

                def heartbeat() -> None:
                    heartbeat_calls['count'] += 1

                def fake_download(_url: str, destination: Path, heartbeat_cb=None) -> None:
                    destination.write_bytes(b'dummy')
                    if heartbeat_cb:
                        heartbeat_cb()

                def fake_extract(_archive: Path, destination: Path, heartbeat_cb=None) -> None:
                    self.assertIsNotNone(heartbeat_cb)
                    if heartbeat_cb:
                        heartbeat_cb()
                    extracted_repo = destination / 'unirig-main'
                    extracted_repo.mkdir(parents=True, exist_ok=True)
                    (extracted_repo / 'run.py').write_text('# stub', encoding='utf-8')

                with patch('generator._download_file', side_effect=fake_download), patch(
                    'generator._extract_archive',
                    side_effect=fake_extract,
                ):
                    generator._download_unirig_repo(runtime, heartbeat_cb=heartbeat)

                self.assertGreaterEqual(heartbeat_calls['count'], 2)

    def test_subprocess_heartbeat_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.runtime_root.mkdir(parents=True, exist_ok=True)
                log_path = runtime.runtime_root / 'bootstrap.log'
                state = generator._default_bootstrap_state(runtime)
                generator._set_state_and_report(runtime, state, None, 10, 'installing official UniRig requirements')
                generator._run_subprocess_with_heartbeat(
                    runtime=runtime,
                    state=state,
                    progress_cb=None,
                    command=[sys.executable, '-c', 'import time; time.sleep(6)'],
                    cwd=runtime.runtime_root,
                    log_path=log_path,
                    phase='installing official UniRig requirements',
                    timeout_sec=30,
                )
                saved = generator._load_bootstrap_state(runtime)
                self.assertFalse(saved['subprocess_alive'])
                self.assertIsNotNone(saved['last_heartbeat_at'])
                self.assertTrue(log_path.exists())

    def test_runtime_status_with_active_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                payload = generator._default_bootstrap_state(runtime)
                payload.update({'install_state': 'installing', 'subprocess_alive': True, 'subprocess_pid': 1234, 'subprocess_cmd': 'pip install -r reqs.txt'})
                generator._write_bootstrap_state(runtime, payload)
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                status = tool.runtime_status()
                self.assertTrue(status['subprocess_alive'])
                self.assertEqual(status['subprocess_pid'], 1234)

    def test_resume_skips_venv_and_torch_when_marked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'run.py').write_text('# stub', encoding='utf-8')
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                generator._mark_phase(runtime, 'create_venv')
                generator._mark_phase(runtime, 'install_torch')
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                with patch('generator._resolve_python311_command', return_value={
                    'command': ['python3.11'],
                    'selected_python': 'python3.11',
                    'selected_python_version': '3.11.9',
                    'selected_python_source': 'PATH_python',
                    'attempts': [],
                    'phases': [],
                }), patch('generator._is_venv_valid', return_value=True), patch('generator._is_module_importable', return_value=True), patch(
                    'generator._bootstrap_runtime',
                    return_value=None,
                ), patch('generator._validate_runtime', return_value={'python': {'major': 3, 'minor': 11}, 'gpu': {'cuda': True}}), patch(
                    'generator._run_subprocess_with_heartbeat',
                    side_effect=AssertionError('should not reinstall venv'),
                ):
                    status = tool.install_runtime()
                self.assertEqual(status['install_state'], 'ready')

    def test_state_persists_between_repair_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                payload = generator._default_bootstrap_state(runtime)
                payload.update({'install_state': 'installing', 'completed_phases': ['prepare_repo', 'create_venv']})
                generator._write_bootstrap_state(runtime, payload)
                first = generator._load_bootstrap_state(runtime)
                second = generator._normalize_state(first, runtime)
                self.assertIn('prepare_repo', second['completed_phases'])
                self.assertIn('create_venv', second['completed_phases'])

    def test_last_error_excerpt_is_saved_on_phase_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {'MODLY_UNIRIG_RUNTIME_DIR': tmp, 'MODLY_UNIRIG_PYTHON311_BIN': sys.executable},
                clear=False,
            ):
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'run.py').write_text('# stub', encoding='utf-8')
                with patch.object(generator.UniRigWorkspaceTool, '_prepare_repo', return_value=False), patch(
                    'generator._resolve_python311_command',
                    return_value={
                        'command': [sys.executable],
                        'selected_python': sys.executable,
                        'selected_python_version': '3.11.9',
                        'selected_python_source': 'PATH_python',
                        'attempts': [],
                        'phases': [],
                    },
                ), patch('generator._is_venv_valid', return_value=False), patch(
                    'generator._run_subprocess_with_heartbeat',
                    side_effect=_WorkspaceToolError('synthetic pip failure'),
                ):
                    with self.assertRaises(_WorkspaceToolError):
                        tool.install_runtime()
                saved = generator._load_bootstrap_state(runtime)
                self.assertEqual(saved['install_state'], 'error')
                self.assertTrue(saved.get('last_error_excerpt'))

    def test_runtime_not_ready_when_lightning_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'run.py').write_text('# stub', encoding='utf-8')
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                with patch('generator._resolve_python311_command', return_value={
                    'command': ['python3.11'],
                    'selected_python': 'python3.11',
                    'selected_python_version': '3.11.9',
                    'selected_python_source': 'PATH_python',
                    'attempts': [],
                    'phases': [],
                }), patch('generator._run_subprocess_with_heartbeat', return_value=None), patch(
                    'generator._bootstrap_runtime',
                    return_value=None,
                ), patch(
                    'generator._validate_runtime',
                    side_effect=_WorkspaceToolError('UniRig runtime import validation failed. Missing modules: lightning'),
                ):
                    with self.assertRaises(_WorkspaceToolError):
                        tool.install_runtime()
                state = generator._load_bootstrap_state(runtime)
                self.assertEqual(state['install_state'], 'error')
                self.assertIn('Missing modules: lightning', state['last_error'])

    def test_validate_runtime_fails_when_runpy_help_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'run.py').write_text('# stub', encoding='utf-8')
                (runtime.repo_dir / 'configs' / 'task').mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / generator.SKELETON_TASK).write_text('', encoding='utf-8')
                (runtime.repo_dir / generator.SKIN_TASK).write_text('', encoding='utf-8')
                (runtime.repo_dir / 'src' / 'inference').mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'src' / 'inference' / 'merge.py').write_text('', encoding='utf-8')
                runtime.python_exe.parent.mkdir(parents=True, exist_ok=True)
                runtime.python_exe.write_text('', encoding='utf-8')
                with patch('generator._run_capture', return_value='{"major":3,"minor":11,"micro":9}\n'), patch(
                    'generator._query_runtime_info',
                    return_value={'cuda': True, 'vram_gb': 16.0},
                ), patch(
                    'generator._query_required_import_validation',
                    return_value={'results': {}, 'missing_modules': []},
                ), patch(
                    'generator._validate_runpy_entrypoint',
                    side_effect=_WorkspaceToolError('run.py entrypoint validation failed: ModuleNotFoundError: lightning'),
                ):
                    with self.assertRaises(_WorkspaceToolError):
                        generator._validate_runtime(runtime)

    def test_repair_detects_missing_required_modules_and_reinstalls_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                runtime = generator._resolve_runtime_context()
                runtime.repo_dir.mkdir(parents=True, exist_ok=True)
                (runtime.repo_dir / 'requirements.txt').write_text('lightning\n', encoding='utf-8')
                state = generator._default_bootstrap_state(runtime)
                with patch('generator._query_missing_required_modules', side_effect=[['lightning'], []]), patch(
                    'generator._run_subprocess_with_heartbeat',
                    return_value=None,
                ) as run_mock:
                    generator._repair_missing_unirig_dependencies(runtime, state, runtime.runtime_root / 'bootstrap.log')
                run_mock.assert_called()
                self.assertEqual(state['missing_required_modules'], [])

    def test_no_custom_optional_core_dependency_split_exists(self) -> None:
        self.assertFalse(hasattr(generator, 'DEFAULT_PYG_OPTIONAL_PACKAGES'))
        self.assertFalse(hasattr(generator, '_split_requirements_groups'))
        self.assertFalse(hasattr(generator, '_install_filtered_requirements'))


if __name__ == '__main__':
    unittest.main()
