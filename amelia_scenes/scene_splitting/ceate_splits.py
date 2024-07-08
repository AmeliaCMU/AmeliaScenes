import os

from easydict import EasyDict

from amelia_scenes.utils import dataset
from amelia_scenes.utils.common import SUPPORTED_AIRPORTS


def run(base_dir: str, traj_version: str, split_type: str, seed: int) -> None:
    traj_data_dir = f"traj_data_{traj_version}"

    config = EasyDict({
        "in_data_dir": os.path.join(base_dir, traj_data_dir, 'proc_full_scenes'),
        "out_data_dir": os.path.join(base_dir, traj_data_dir),
        "seed": seed,
        "split_type": split_type,
        "random_splits": {
            "train_val_test": [0.7, 0.1, 0.2],
            "unseen_perc": 0.25
        },
        "frame_splits": {
            "train_val_test": [0.7, 0.1, 0.2],
            "train_val_perc": 0.75,
            "unseen_perc": 0.25
        },
    })

    if split_type == "random":
        # Will split data randomly. Simplest one, but potential data leakage.
        dataset.create_random_splits(config, ['kbtp'])
    elif split_type == 'frame':
        # Will split data by frames.
        dataset.create_frame_splits(config, ['kbtp'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="../../datasets/amelia")
    parser.add_argument("--traj_version", type=str, default="a10v08")
    parser.add_argument("--split_type", default='random',
                        choices=['random', 'frame'])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(**vars(args))
