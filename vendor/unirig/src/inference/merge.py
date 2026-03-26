from __future__ import annotations

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
    if source.suffix.lower() != '.glb':
        raise SystemExit(
            'Offline fallback merge only supports GLB sources; '
            f"got '{source.suffix or '<no extension>'}' from {source}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
