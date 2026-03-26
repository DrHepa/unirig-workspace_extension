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
                }), patch('generator._create_venv', return_value=None), patch('generator._bootstrap_runtime', return_value=None), patch(
                    'generator._validate_runtime', return_value={'python': {'major': 3, 'minor': 11}, 'gpu': {'cuda': True}}
                ):
                    status = tool.install_runtime(progress_cb=lambda pct, step: progress.append((pct, step)))
                self.assertEqual(status['install_state'], 'ready')
                saved = generator._load_bootstrap_state(runtime)
                self.assertEqual(saved['install_state'], 'ready')
                self.assertEqual(saved['step'], 'ready')
                self.assertGreaterEqual(saved['percent'], 100)
                self.assertTrue(progress)


if __name__ == '__main__':
    unittest.main()
