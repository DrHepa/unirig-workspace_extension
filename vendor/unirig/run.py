import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniRig CLI (offline fallback)')
    parser.add_argument('--task')
    parser.add_argument('--seed')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--npz_dir')
    parser.add_argument('--data_name')
    parser.parse_args()
