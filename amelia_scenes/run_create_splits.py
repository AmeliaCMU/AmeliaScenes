import os

from easydict import EasyDict

from amelia_scenes.utils import dataset
from amelia_scenes.utils.common import SUPPORTED_AIRPORTS, ROOT_DIR


def run(base_dir: str, traj_version: str, split_type: str, seed: int) -> None:
    traj_data_dir = f"traj_data_{traj_version}"
    config = EasyDict({
        "in_data_dir": os.path.join(base_dir, traj_data_dir, 'proc_full_scenes'),
        "out_data_dir": os.path.join(base_dir, traj_data_dir, 'splits'),
        "seed": seed,
        "split_type": split_type,
        "random_splits": {
            "train_val_test": [0.7, 0.1, 0.2],
            "unseen_perc": 0.25
        },
        "day_splits": {
            "train_val_test": [0.7, 0.1, 0.2],
            "train_val_perc": 0.75,
            "unseen_perc": 0.25
        },
        "month_splits": {
            "train_val_test": [0.7, 0.1, 0.2],
            "train_val_perc": 0.75,
            "unseen_perc": 0.25
        },
    })

    # get airport data
    airport_list = [
        airport for airport in SUPPORTED_AIRPORTS if os.path.exists(os.path.join(config.in_data_dir, airport))]

    if split_type == "random":
        # Will split data randomly. Simplest one, but potential data leakage.
        dataset.create_random_splits(config, airport_list)
    elif split_type == 'day':
        # Will split data by day. Useful to ensure that different days go into the train/test sets to avoid data leakage
        dataset.create_day_splits(config, airport_list)
    else:  # month
        # Will split data by month. Useful if we're training multiple months for a single airport, but can handle single-month airports as well.
        dataset.create_month_splits(config, airport_list)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default=f"{ROOT_DIR}/datasets/amelia")
    parser.add_argument("--traj_version", type=str, default="a10v08")
    parser.add_argument("--split_type", default='random',
                        choices=['random', 'day', 'month'])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(**vars(args))
