from __future__ import annotations

import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

UPSTREAM_REPO = 'VAST-AI-Research/UniRig'
UPSTREAM_REF = 'main'
UPSTREAM_ZIP_URL = f'https://github.com/{UPSTREAM_REPO}/archive/{UPSTREAM_REF}.zip'

ROOT = Path(__file__).resolve().parent
VENDOR_DIR = ROOT / 'vendor'
UNIRIG_DIR = VENDOR_DIR / 'unirig'

UNIRIG_REQUIRED = [
    'run.py',
    'src',
    'configs',
    'inference',
    'evaluation',
]

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


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def rebuild_vendor() -> None:
    if VENDOR_DIR.exists():
        shutil.rmtree(VENDOR_DIR)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='unirig_vendor_') as tmp:
        tmp_root = Path(tmp)
        zip_path = tmp_root / f'unirig-{UPSTREAM_REF}.zip'
        urllib.request.urlretrieve(UPSTREAM_ZIP_URL, str(zip_path))

        unpack_dir = tmp_root / 'unpack'
        shutil.unpack_archive(str(zip_path), str(unpack_dir))
        source_root = next(p for p in unpack_dir.iterdir() if p.is_dir())

        UNIRIG_DIR.mkdir(parents=True, exist_ok=True)
        for rel in UNIRIG_REQUIRED:
            src = source_root / rel
            dst = UNIRIG_DIR / rel
            if not src.exists():
                raise FileNotFoundError(f'Missing upstream path: {src}')
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    run(['python', '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
    run(['python', '-m', 'pip', 'install', '--target', str(VENDOR_DIR), *PURE_PYTHON_VENDOR])


if __name__ == '__main__':
    rebuild_vendor()
    print(f'vendor ready at: {VENDOR_DIR}')
