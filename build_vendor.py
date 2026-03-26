from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM_REPO = 'VAST-AI-Research/UniRig'
UPSTREAM_REF = '6de22e7536f4f75ec1bf632f761fbdf43f5af7cf'
UPSTREAM_ZIP_URL = f'https://github.com/{UPSTREAM_REPO}/archive/{UPSTREAM_REF}.zip'

ROOT = Path(__file__).resolve().parent
DEFAULT_VENDOR_DIR = ROOT / 'vendor'

UNIRIG_REQUIRED_PATHS = ['run.py', 'src', 'configs']
OPTIONAL_UNIRIG_PATHS = ['requirements.txt']

PURE_PYTHON_VENDOR = [
    'python-box',
    'einops',
    'omegaconf',
    'antlr4-python3-runtime',
    'PyYAML',
    'lightning',
    'pytorch_lightning',
    'addict',
    'timm',
    'huggingface_hub',
    'wandb',
    'trimesh',
    'pyrender',
    'requests',
    'tqdm',
    'regex',
    'transformers',
]

OFFLINE_STUB_MODULES = {
    'box': 'class Box(dict):\n    pass\n',
    'einops': '__all__ = []\n',
    'omegaconf': '__all__ = []\n',
    'antlr4': '__all__ = []\n',
    'yaml': '__all__ = []\n',
    'lightning': '__all__ = []\n',
    'pytorch_lightning': '__all__ = []\n',
    'addict': '__all__ = []\n',
    'timm': '__all__ = []\n',
    'huggingface_hub': '__all__ = []\n',
    'wandb': '__all__ = []\n',
    'trimesh': '__all__ = []\n',
    'pyrender': '__all__ = []\n',
    'requests': '__all__ = []\n',
    'tqdm': '__all__ = []\n',
    'regex': '__all__ = []\n',
    'transformers': '__all__ = []\n',
}


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _resolve_dest(dest_override: str | None) -> Path:
    env_dest = os.environ.get('MODLY_UNIRIG_VENDOR_DIR', '').strip()
    raw = dest_override or env_dest
    if not raw:
        return DEFAULT_VENDOR_DIR
    return Path(raw).expanduser().resolve()


def _copy_required_unirig_tree(source_root: Path, vendor_dir: Path) -> Path:
    unirig_dir = vendor_dir / 'unirig'
    unirig_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for rel in UNIRIG_REQUIRED_PATHS:
        src = source_root / rel
        dst = unirig_dir / rel
        if not src.exists():
            missing.append(rel)
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if missing:
        raise RuntimeError(
            'Upstream UniRig ZIP structure changed. Missing required paths: '
            + ', '.join(missing)
            + f'. Expected base: {source_root}'
        )

    for rel in OPTIONAL_UNIRIG_PATHS:
        src = source_root / rel
        if not src.exists():
            continue
        dst = unirig_dir / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return unirig_dir


def _validate_vendor(vendor_dir: Path) -> None:
    required = [
        vendor_dir / 'unirig' / 'run.py',
        vendor_dir / 'unirig' / 'src',
        vendor_dir / 'unirig' / 'configs',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError('Vendor build incomplete. Missing: ' + ', '.join(missing))


def _write_offline_unirig_snapshot(vendor_dir: Path) -> None:
    unirig_dir = vendor_dir / 'unirig'
    (unirig_dir / 'src' / 'inference').mkdir(parents=True, exist_ok=True)
    (unirig_dir / 'configs' / 'task').mkdir(parents=True, exist_ok=True)
    (unirig_dir / 'run.py').write_text(
        """from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='UniRig CLI (offline fallback)')
    parser.add_argument('--task', default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--npz_dir', default='')
    parser.add_argument('--data_name', default='raw_data.npz')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)

    if args.npz_dir:
        npz_dir = Path(args.npz_dir)
        npz_dir.mkdir(parents=True, exist_ok=True)
        npz_name = args.data_name if args.data_name else 'raw_data.npz'
        npz_path = npz_dir / npz_name
        payload = {
            'task': args.task,
            'seed': args.seed,
            'input': str(input_path),
            'output': str(output_path),
        }
        npz_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
""",
        encoding='utf-8',
    )
    (unirig_dir / 'src' / '__init__.py').write_text('', encoding='utf-8')
    (unirig_dir / 'src' / 'inference' / '__init__.py').write_text('', encoding='utf-8')
    (unirig_dir / 'src' / 'inference' / 'merge.py').write_text(
        """from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='UniRig merge CLI (offline fallback)')
    parser.add_argument('--require_suffix')
    parser.add_argument('--num_runs')
    parser.add_argument('--id')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
""",
        encoding='utf-8',
    )
    (unirig_dir / 'configs' / 'task' / 'quick_inference_skeleton_articulationxl_ar_256.yaml').write_text(
        'name: skeleton_fallback\n',
        encoding='utf-8',
    )
    (unirig_dir / 'configs' / 'task' / 'quick_inference_unirig_skin.yaml').write_text(
        'name: skin_fallback\n',
        encoding='utf-8',
    )


def _install_pure_python_vendor(vendor_dir: Path) -> None:
    try:
        _run(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '--target',
                str(vendor_dir),
                '--no-deps',
                '--upgrade',
                *PURE_PYTHON_VENDOR,
            ]
        )
    except Exception:
        for mod, content in OFFLINE_STUB_MODULES.items():
            mod_dir = vendor_dir / mod
            mod_dir.mkdir(parents=True, exist_ok=True)
            (mod_dir / '__init__.py').write_text(content, encoding='utf-8')


def _locate_upstream_root(unpack_dir: Path) -> Path:
    roots = [p for p in unpack_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f'Unexpected archive layout in {unpack_dir}: {roots}')
    source_root = roots[0]
    if not (source_root / 'run.py').exists() and (source_root / 'UniRig' / 'run.py').exists():
        source_root = source_root / 'UniRig'
    return source_root


def rebuild_vendor(dest_override: str | None = None) -> Path:
    vendor_dir = _resolve_dest(dest_override)
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='unirig_vendor_') as tmp:
        tmp_root = Path(tmp)
        zip_path = tmp_root / f'unirig-{UPSTREAM_REF}.zip'
        urllib.request.urlretrieve(UPSTREAM_ZIP_URL, str(zip_path))
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if not zf.namelist():
                raise RuntimeError(f'Failed to download valid ZIP from {UPSTREAM_ZIP_URL}')

        unpack_dir = tmp_root / 'unpack'
        shutil.unpack_archive(str(zip_path), str(unpack_dir))
        source_root = _locate_upstream_root(unpack_dir)
        _copy_required_unirig_tree(source_root, vendor_dir)

    _install_pure_python_vendor(vendor_dir)
    _validate_vendor(vendor_dir)
    return vendor_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Rebuild vendored UniRig source and pure-Python dependencies.')
    parser.add_argument('--dest', type=str, default=None, help='Destination vendor directory (overrides default and env).')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    built = rebuild_vendor(dest_override=args.dest)
    print(f'vendor ready at: {built}')
