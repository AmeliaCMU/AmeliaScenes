import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import random
import seaborn as sns

from natsort import natsorted
from tqdm import tqdm

from amelia_scenes.utils.dataset import load_assets
from amelia_scenes.utils.common import SUPPORTED_AIRPORTS, ROOT_DIR
from amelia_scenes.visualization import scene_viz as viz
from amelia_scenes.visualization.scene_viz import SUPPORTED_SCENES_TYPES

def run(
    airport: str,
    base_path: str,
    traj_version: str,
    out_path: str,
    num_scenes: int,
    perc: float,
    seed: int,
    dpi: int
):
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
    else:
        scene_files = scene_files[:int(len(scene_files) * perc)]

    out_dir = os.path.join(out_path, airport)
    os.makedirs(out_dir, exist_ok=True)
    score_file = os.path.join(out_dir, 'scores.csv')
    if not os.path.exists(score_file):
        scores = {
            'filename': [],
            'crowdedness': [],
            'kinematic': [],
            'interactive': [],
            'critical': []
        }
        for scene_file in tqdm(scene_files):
            with open(scene_file, 'rb') as f:
                scene = pickle.load(f)
            if not scene['meta']:
                continue

            # fsplit = scene_file.split('/')
            # scenario_name = f"{fsplit[-2]}/{fsplit[-1]}"
            # scores['filename'].append(scenario_name)
            scores['filename'].append(scene_file)
            scores['crowdedness'].append(scene['meta']['scene_scores']['crowdedness'])
            scores['kinematic'].append(scene['meta']['scene_scores']['kinematic'])
            scores['interactive'].append(scene['meta']['scene_scores']['interactive'])
            scores['critical'].append(scene['meta']['scene_scores']['critical'])
        
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(score_file, index=False)
    else:
        scores_df = pd.read_csv(score_file)

    for key in scores_df.keys():
        if key == 'filename':
            continue    
        sns.histplot(data=scores_df, x=key, kde=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{key}_hist.png'), dpi=dpi)
        plt.close()

        sns.kdeplot(
            data=scores_df[key], label=key, linewidth=2, bw_adjust=5, common_norm=False, cut=0, 
            fill=True, alpha=0.15)
        plt.xlabel('Scores')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(out_dir, f'{key}_kde.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # TODO: save percentiles and plot scenes in the top 10 and bottom 10 percentiles

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
        "--out_path",
        type=str,
        default=f"{ROOT_DIR}/out/scores",
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
