from __future__ import annotations

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
