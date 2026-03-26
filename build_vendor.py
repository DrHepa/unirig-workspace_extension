from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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


def rebuild_vendor(dest_override: str | None = None) -> Path:
    vendor_dir = _resolve_dest(dest_override)
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='unirig_vendor_') as tmp:
        tmp_root = Path(tmp)
        zip_path = tmp_root / f'unirig-{UPSTREAM_REF}.zip'
        urllib.request.urlretrieve(UPSTREAM_ZIP_URL, str(zip_path))

        unpack_dir = tmp_root / 'unpack'
        shutil.unpack_archive(str(zip_path), str(unpack_dir))
        roots = [p for p in unpack_dir.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f'Unexpected archive layout in {unpack_dir}: {roots}')
        _copy_required_unirig_tree(roots[0], vendor_dir)

    _run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    _run([sys.executable, '-m', 'pip', 'install', '--target', str(vendor_dir), *PURE_PYTHON_VENDOR])
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
