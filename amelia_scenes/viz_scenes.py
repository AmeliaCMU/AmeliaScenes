import os
import pickle
import random

from natsort import natsorted
from tqdm import tqdm

from amelia_scenes.utils.dataset import load_assets
from amelia_scenes.utils.common import SUPPORTED_AIRPORTS
from amelia_scenes.visualization import scene_viz as viz

SUBDIR = __file__.split('/')[-1].split('.')[0]

def run(
    airport: str,
    base_path: str, 
    traj_version: str,
    out_path: str, 
    num_scenes: int, 
    perc: float, 
    benchmark: bool,
    seed: int,
    dpi: int
):
    assets = load_assets(base_path, airport)
    traj_data_dir = "traj_data_benchmark" if benchmark else f"traj_data_{traj_version}" 

    scenes_dir = os.path.join(base_path, traj_data_dir, 'proc_full_scenes', airport)
    scenes_subdirs = [os.path.join(scenes_dir, sdir) for sdir in os.listdir(scenes_dir)]
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
    
    out_dir = os.path.join(out_path, SUBDIR, airport)
    os.makedirs(out_dir, exist_ok=True)

    for scene_file in tqdm(scene_files):
        with open(scene_file, 'rb') as f:
            scene = pickle.load(f)
        
        fsplit = scene_file.split('/')
        scenario_name, scenario_id = fsplit[-2], fsplit[-1].split('.')[0]
        filetag = os.path.join(out_dir, f"{scenario_name}_{scenario_id}.png")
        viz.plot_scene(scene, assets, filetag, dpi=dpi)

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
        default="../datasets/amelia", 
        help="Path to dataset to visualize.")
    parser.add_argument(
        "--traj_version", 
        type=str, 
        default="a10v08")
    parser.add_argument(
        "--out_path", 
        type=str, 
        default="../out/vis", 
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