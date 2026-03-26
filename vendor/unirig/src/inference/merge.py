import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniRig merge CLI (offline fallback)')
    parser.add_argument('--require_suffix')
    parser.add_argument('--num_runs')
    parser.add_argument('--id')
    parser.add_argument('--source')
    parser.add_argument('--target')
    parser.add_argument('--output')
    parser.parse_args()
