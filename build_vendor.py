from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UPSTREAM_REPO = 'VAST-AI-Research/UniRig'
DEFAULT_REF = os.environ.get('MODLY_UNIRIG_REPO_REF', 'main')
ROOT = Path(__file__).resolve().parent
DEFAULT_VENDOR_DIR = ROOT / 'vendor'

REQUIRED_PATHS = ['run.py', 'src', 'configs', 'requirements.txt']
OPTIONAL_PATHS = ['launch', 'blender']


def _archive_url_for_ref(ref: str) -> str:
    ref = (ref or 'main').strip()
    if len(ref) >= 7 and all(c in '0123456789abcdefABCDEF' for c in ref):
        return f'https://github.com/{UPSTREAM_REPO}/archive/{ref}.zip'
    return f'https://github.com/{UPSTREAM_REPO}/archive/refs/heads/{ref}.zip'


def _resolve_source(source_override: str | None, ref: str | None) -> str:
    env_source = os.environ.get('MODLY_UNIRIG_SOURCE_ZIP', '').strip()
    raw = (source_override or env_source or '').strip()
    if raw:
        return raw
    return _archive_url_for_ref(ref or DEFAULT_REF)


def _resolve_dest(dest_override: str | None) -> Path:
    env_dest = os.environ.get('MODLY_UNIRIG_VENDOR_DIR', '').strip()
    raw = (dest_override or env_dest or '').strip()
    if not raw:
        return DEFAULT_VENDOR_DIR
    return Path(raw).expanduser().resolve()


def _fetch_archive(source: str, dst_zip: Path) -> None:
    source_path = Path(source)
    if source_path.exists():
        shutil.copy2(source_path, dst_zip)
        return
    urllib.request.urlretrieve(source, str(dst_zip))


def _locate_upstream_root(unpack_dir: Path) -> Path:
    roots = [p for p in unpack_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f'Unexpected archive layout in {unpack_dir}: {[p.name for p in roots]}')
    root = roots[0]
    if not (root / 'run.py').exists() and (root / 'UniRig' / 'run.py').exists():
        root = root / 'UniRig'
    return root


def _copy_tree(source_root: Path, vendor_dir: Path) -> None:
    unirig_dir = vendor_dir / 'unirig'
    unirig_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for rel in REQUIRED_PATHS:
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
            + f'. Source root: {source_root}'
        )

    for rel in OPTIONAL_PATHS:
        src = source_root / rel
        if not src.exists():
            continue
        dst = unirig_dir / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _validate_vendor(vendor_dir: Path) -> None:
    required = [vendor_dir / 'unirig' / rel for rel in REQUIRED_PATHS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError('Vendor build incomplete. Missing: ' + ', '.join(missing))


def rebuild_vendor(dest_override: str | None = None, source_override: str | None = None, ref_override: str | None = None) -> Path:
    vendor_dir = _resolve_dest(dest_override)
    if vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)

    source = _resolve_source(source_override, ref_override)
    with tempfile.TemporaryDirectory(prefix='unirig_vendor_') as tmp:
        tmp_root = Path(tmp)
        zip_path = tmp_root / 'unirig.zip'
        _fetch_archive(source, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if not zf.namelist():
                raise RuntimeError(f'Failed to read UniRig ZIP from: {source}')
        unpack_dir = tmp_root / 'unpack'
        shutil.unpack_archive(str(zip_path), str(unpack_dir))
        source_root = _locate_upstream_root(unpack_dir)
        _copy_tree(source_root, vendor_dir)

    _validate_vendor(vendor_dir)
    return vendor_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Download and vendor the UniRig source tree required for runtime inference.')
    parser.add_argument('--dest', default=None, help='Destination vendor directory. Defaults to ./vendor or MODLY_UNIRIG_VENDOR_DIR.')
    parser.add_argument('--source-zip', default=None, help='Local path or URL for a UniRig ZIP archive. Defaults to MODLY_UNIRIG_SOURCE_ZIP or the pinned ref URL.')
    parser.add_argument('--ref', default=None, help='Git ref to resolve when --source-zip is not provided. Defaults to MODLY_UNIRIG_REPO_REF or main.')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    built = rebuild_vendor(dest_override=args.dest, source_override=args.source_zip, ref_override=args.ref)
    print(f'vendor ready at: {built}')
