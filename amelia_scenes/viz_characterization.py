import os
import pickle
import random
import numpy as np

from natsort import natsorted
from tqdm import tqdm

from amelia_scenes.utils.dataset import load_assets
from amelia_scenes.utils.common import SUPPORTED_AIRPORTS, ROOT_DIR, EPS
from amelia_scenes.scoring.kinematic import compute_kinematic_features, compute_kinematic_scores
from amelia_scenes.scoring.interactive import compute_interactive_features, compute_interactive_scores
from amelia_scenes.visualization import scene_viz as viz

SUBDIR = __file__.split('/')[-1].split('.')[0]


def run(
    airport: str,
    base_path: str,
    traj_version: str,
    graph_version: str,
    feature_type: str,
    out_path: str,
    num_scenes: int,
    perc: float,
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
    else:
        scene_files = scene_files[:int(len(scene_files) * perc)]

    out_dir = os.path.join(out_path, SUBDIR, airport, f'{feature_type}_features')
    os.makedirs(out_dir, exist_ok=True)

    for scene_file in tqdm(scene_files):
        with open(scene_file, 'rb') as f:
            scene = pickle.load(f)

        # Compute metrics
        hold_lines = assets[1][:, 2:4]
        if feature_type == 'individual':
            features = compute_kinematic_features(scene, hold_lines)
            scores = compute_kinematic_scores(scene, assets[1][:, 2:4], features=features)
        elif feature_type == 'interactive':
            features_full = compute_interactive_features(scene, hold_lines)
            N = scene['agent_sequences'].shape[0]

            features = {'mttcp': np.zeros(N), 'collisions': np.zeros(N), 'agent_idxs': []}
            for n, (i, j) in enumerate(features_full['agent_ids']):
                if i not in features['agent_idxs']:
                    features['agent_idxs'].append(i)
                if j not in features['agent_idxs']:
                    features['agent_idxs'].append(j)
                features['mttcp'][i] += 1/(features_full['mttcp'][n]+EPS)
                features['collisions'][i] += features_full['collisions'][n]
                
                features['mttcp'][j] += 1/(features_full['mttcp'][n]+EPS)
                features['collisions'][j] += features_full['collisions'][n]

            scores = compute_interactive_scores(scene, hold_lines, features=features_full)
        else:
            raise ValueError(f"Feature type '{feature_type}' not supported.")       
        
        fsplit = scene_file.split('/')
        scenario_name, scenario_id = fsplit[-2], fsplit[-1].split('.')[0]
        filetag = os.path.join(out_dir, f"{scenario_name}_{scenario_id}.png")
        viz.plot_scene(scene, assets, filetag, 'features', dpi=dpi, features=features)

        scores_dict = {
            'agent_scores': scores[0], 
            'scene_score': scores[1], 
            'valid_agents': features['agent_idxs']
        }
        filetag = os.path.join(out_dir, f"{scenario_name}_{scenario_id}_scores.png")
        viz.plot_scene(scene, assets, filetag, 'scores', dpi=dpi, scores=scores_dict)


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
        "--feature_type",
        default='individual',
        choices=['individual', 'interactive'],)
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
