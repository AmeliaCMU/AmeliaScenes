import os
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
    perc: float,
    viz_scene: str,
    scene_id: str,
    benchmark: bool,
    xplane: bool,
    scene_type: str,
    show_scores: bool,
    seed: int,
    dpi: int,
    to_scale: bool
):
    assets = load_assets(base_path, airport, graph_file=f'graph_data_{graph_version}')
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
        if viz_scene:
            # check if scene id is provided
            file_name = viz_scene
            if scene_id:
                file_name = os.path.join(viz_scene, f"{scene_id}")

            scene_files = [scene_file for scene_file in scene_files if file_name in scene_file]
            if not scene_files:
                raise FileNotFoundError(f"Scene {viz_scene}_{scene_id} not found in {airport} scenes.")
        elif num_scenes > 0:
            scene_files = scene_files[:num_scenes]
        else:
            scene_files = scene_files[:int(len(scene_files) * perc)]

    out_dir = os.path.join(out_path, SUBDIR, airport, scene_type)
    os.makedirs(out_dir, exist_ok=True)

    for scene_file in tqdm(scene_files):
        with open(scene_file, 'rb') as f:
            scene = pickle.load(f)

        # NOTE: wrap for plotting scores
        scores = {}
        scores['agent_scores'] = scene["meta"]['agent_scores']['critical']
        scores['scene_scores'] = scene["meta"]['scene_scores']['critical']
        scores['valid_agents'] = scene['agent_valid']

        fsplit = scene_file.split('/')
        scenario_name, scenario_id = fsplit[-2], fsplit[-1].split('.')[0]
        filetag = os.path.join(out_dir, f"{scenario_name}_{scenario_id}.png")
        viz.plot_scene(scene,
                       assets,
                       filetag,
                       scene_type,
                       dpi=dpi,
                       scores=scores,
                       show_scores=show_scores,
                       to_scale=to_scale)


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
        default="a42v01")
    parser.add_argument(
        "--graph_version",
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
        "--viz_scene",
        type=str,
        default="",
        help="Path to a scene file to visualize. If provided, other arguments are ignored."
        " If not provided, the percentage of scenes to be visualized.")
    parser.add_argument(
        "--scene_id",
        type=str,
        default="",
        help="ID of the scene to visualize. If provided, viz_scene must be provided.")
    parser.add_argument(
        "--benchmark",
        action='store_true')
    parser.add_argument(
        "--xplane",
        action='store_true')
    parser.add_argument(
        "--scene_type",
        default='simple',
        choices=SUPPORTED_SCENES_TYPES)
    parser.add_argument(
        "--show-scores",
        action='store_true',
        help="Show scores in the scene visualization, if the scene type is equal to 'scores'.",
        default=False)
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
    parser.add_argument(
        "--to_scale",
        action='store_true',
        help="Scale agents to the scene limits.",
        default=False)
    args = parser.parse_args()
    run(**vars(args))
