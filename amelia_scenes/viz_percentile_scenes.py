import os
import pandas as pd
import pickle
import random

from natsort import natsorted
from tqdm import tqdm

from amelia_scenes.utils.dataset import load_assets
from amelia_scenes.utils.common import SUPPORTED_AIRPORTS, ROOT_DIR
from amelia_scenes.visualization import scene_viz as viz
from amelia_scenes.visualization.scene_viz import SUPPORTED_SCENES_TYPES

SUBDIR = __file__.split('/')[-1].split('.')[0]


def run(
    airport: str,
    base_path: str,
    traj_version: str,
    graph_version: str,
    out_path: str,
    num_scenes: int,
    score_type: str,
    scene_type: str,
    seed: int,
    dpi: int
):
    assets = load_assets(base_path, airport, graph_file=f'graph_data_{graph_version}')
    traj_data_dir = f"traj_data_{traj_version}" 

    scenes_dir = os.path.join(base_path, traj_data_dir, 'proc_full_scenes', airport)
    scenes_subdirs = [
        os.path.join(scenes_dir, sdir) for sdir in os.listdir(scenes_dir)
        if os.path.isdir(os.path.join(scenes_dir, sdir))
    ]
    scene_files = []
    for subdir in scenes_subdirs:
        scene_files += [os.path.join(subdir, f) for f in natsorted(os.listdir(subdir))]
   
    random.seed(seed)
    random.shuffle(scene_files)
    if num_scenes > 0:
        scene_files = scene_files[:num_scenes]

    percentile_file = os.path.join(out_path, 'scores', airport, f"{score_type}_percentiles.csv")
    assert os.path.exists(percentile_file), f"Percentile file {percentile_file} does not exist."
    percentile_scores = pd.read_csv(percentile_file)
    perc_lo, score_lo = percentile_scores.min().values
    perc_hi, score_hi = percentile_scores.max().values

    out_dir_lo = os.path.join(
        out_path, 'vis', SUBDIR, airport, f"{scene_type}_{score_type}_{perc_lo}")
    os.makedirs(out_dir_lo, exist_ok=True)

    out_dir_hi = os.path.join(
        out_path, 'vis', SUBDIR, airport, f"{scene_type}_{score_type}_{perc_hi}")
    os.makedirs(out_dir_hi, exist_ok=True)

    for scene_file in tqdm(scene_files):
        with open(scene_file, 'rb') as f:
            scene = pickle.load(f)
        if not scene['meta']:
            continue
        score = scene['meta']['scene_scores'][score_type]
        if score <= score_lo:
            fsplit = scene_file.split('/')
            scenario_name, scenario_id = fsplit[-2], fsplit[-1].split('.')[0]
            filetag = os.path.join(out_dir_lo, f"{scenario_name}_{scenario_id}_score-{score}.png")
            viz.plot_scene(scene, assets, filetag, scene_type, dpi=dpi)
        elif score >= score_hi:
            fsplit = scene_file.split('/')
            scenario_name, scenario_id = fsplit[-2], fsplit[-1].split('.')[0]
            filetag = os.path.join(out_dir_hi, f"{scenario_name}_{scenario_id}_score-{score}.png")
            viz.plot_scene(scene, assets, filetag, scene_type, dpi=dpi)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--airport",
        type=str,
        default="kbos",
        choices=SUPPORTED_AIRPORTS)
    parser.add_argument(
        "--base_path",
        type=str,
        default=f"{ROOT_DIR}/datasets/amelia",
        help="Path to dataset to visualize.")
    parser.add_argument(
        "--traj_version",
        type=str,
        default="a10v08")
    parser.add_argument(
        "--graph_version",
        type=str,
        default="a10v01os")
    parser.add_argument(
        "--out_path",
        type=str,
        default=f"{ROOT_DIR}/out",
        help="Output path.")
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=-1,
        help="Scenes to process.")
    parser.add_argument(
        "--score_type",
        type=str,
        default='crowdedness',
        choices=['crowdedness', 'kinematic', 'interactive', 'critical'],
        help="Scene percentile to visualize.")
    parser.add_argument(
        "--scene_type",
        default='simple',
        choices=SUPPORTED_SCENES_TYPES)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.")
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Random seed.")
    args = parser.parse_args()
    run(**vars(args))
