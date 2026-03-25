import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
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
    def test_runtime_status_without_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'MODLY_UNIRIG_RUNTIME_DIR': tmp}, clear=False):
                tool = generator.UniRigWorkspaceTool(Path(tmp))
                status = tool.runtime_status()
                self.assertEqual(status['install_state'], 'not_installed')
                self.assertEqual(status['percent'], 0)
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
                with self.assertRaises(_WorkspaceToolError):
                    tool.install_runtime()
                state_path = Path(tmp) / 'bootstrap_state.json'
                state = json.loads(state_path.read_text(encoding='utf-8'))
                self.assertEqual(state['install_state'], 'error')
                self.assertIn('Python 3.11', state['last_error'])

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


if __name__ == '__main__':
    unittest.main()
