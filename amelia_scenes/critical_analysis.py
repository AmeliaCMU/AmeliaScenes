import os
import numpy as np
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
    out_path: str,
    num_scenes: int,
    perc: float,
    benchmark: bool,
    xplane: bool,
    scene_type: str,
    seed: int,
    dpi: int
):
    assets = load_assets(base_path, airport)
    tag = 'benchmark' if benchmark else 'xplane' if xplane else traj_version
    traj_data_dir = f"traj_data_{tag}"

    scenes_dir = os.path.join(base_path, traj_data_dir, 'proc_full_scenes', airport)
    scenes_subdirs = [
        os.path.join(scenes_dir, sdir) for sdir in os.listdir(scenes_dir)
        if os.path.isdir(os.path.join(scenes_dir, sdir))
    ]
    scene_files = []
    for subdir in scenes_subdirs:
        scene_files += [os.path.join(subdir, f) for f in natsorted(os.listdir(subdir))]

    if not benchmark:
        random.seed(seed)
        random.shuffle(scene_files)
        if num_scenes > 0:
            scene_files = scene_files[:num_scenes]
        else:
            scene_files = scene_files[:int(len(scene_files) * perc)]

    out_dir = os.path.join(out_path, SUBDIR, airport, scene_type)
    os.makedirs(out_dir, exist_ok=True)

    scores_list = []
    scenes_data = []

    for scene_file in tqdm(scene_files):
        with open(scene_file, 'rb') as f:
            scene = pickle.load(f)
        # Extract score from scene; adjust based on your scene structure
        score = scene["meta"]["scene_scores"]["critical"]

        if score is not None:
            scores_list.append(score)
            scenes_data.append((scene_file, scene, score))
        else:
            print(f"Score not found in {scene_file}")

    # Compute the single percentile value (e.g., the 50th percentile for the median)
    percentile = 90
    single_percentile = np.percentile(scores_list, percentile)
    print(f"The {percentile} percentile score is:", single_percentile)

    scenes_in_percentile = [(scene_file, scene)
                            for scene_file, scene, score in scenes_data
                            if score >= single_percentile]

    print(f"Found {len(scenes_in_percentile)}")
    for scene_file, scene in scenes_in_percentile:
        fsplit = scene_file.split('/')
        scenario_name, scenario_id = fsplit[-2], fsplit[-1].split('.')[0]
        # breakpoint()
        id = int(scenario_id.split("_")[0])
        if id > 250 or id < 220:
            continue
        filetag = os.path.join(out_dir, f"{scenario_name}_{scenario_id}.png")
        viz.plot_scene(scene, assets, filetag, scene_type, dpi=200, scores=None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--airport",
        type=str,
        default="khou",
        choices=SUPPORTED_AIRPORTS)
    parser.add_argument(
        "--base_path",
        type=str,
        default=f"{ROOT_DIR}/datasets/amelia",
        help="Path to dataset to visualize.")
    parser.add_argument(
        "--traj_version",
        type=str,
        default="a42v01")
    parser.add_argument(
        "--out_path",
        type=str,
        default=f"{ROOT_DIR}/out/vis",
        help="Output path.")
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=-1,
        help="Scenes to process. Alternative, use --perc.")
    parser.add_argument(
        "--perc",
        type=float,
        default=1.0,
        help="Percentage of files to load (0.0, 1]. Alternative, use --num_scenes.")
    parser.add_argument(
        "--benchmark",
        action='store_true')
    parser.add_argument(
        "--xplane",
        action='store_true')
    parser.add_argument(
        "--scene_type",
        default='scores',
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
