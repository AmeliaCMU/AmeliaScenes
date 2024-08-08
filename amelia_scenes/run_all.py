from run_individual import compute_metrics as compute_individual_metrics
from run_interaction import compute_metrics as compute_interaction_metrics
from amelia_scenes.scene_utils.common import Status as s
import amelia_scenes.scene_utils.common as C
import amelia_scenes.scene_utils.scores as S
from tqdm import tqdm
import glob
import numpy as np
import os
import pickle
import random
random.seed(42)
np.set_printoptions(suppress=True)

from amelia_scenes.utils.common import SUPPORTED_AIRPORTS


def run(airport: str, base_dir: str, out_dir: str, plot: bool, num_scenes: int):
    
    os.makedirs(args.out_dir, exist_ok=True)

    scenarios_dir = os.path.join(args.base_dir, "proc_trajectories", args.airport)
    map_dir = os.path.join(args.base_dir, "maps", args.airport)

    assets = C.load_assets(map_dir=map_dir)

    scenarios = glob.glob(f"{scenarios_dir}/**/*.pkl", recursive=True)
    random.shuffle(scenarios)
    num_scenes = args.num_scenes
    if num_scenes > 0:
        scenarios = scenarios[:num_scenes]

    # Hold line locations
    raster_map, hold_lines, graph_map, ll_extent, agents = assets
    hl_xy = hold_lines[:, 2:4]
    rwy_ext = C.RUNWAY_EXTENTS[args.airport]["max"]

    num_noninteractive, num_nonaircraft = 0, 0
    for n, scenario in enumerate(tqdm(scenarios)):
        split = scenario.split("/")
        subdir, scenario_id = split[-2], split[-1].split('.')[0]

        outdir = os.path.join(args.out_dir, "highest_scores", args.airport)
        os.makedirs(outdir, exist_ok=True)

        with open(scenario, 'rb') as f:
            scene = pickle.load(f)

        agent_types = np.asarray(scene['agent_types'])
        if len(np.where(agent_types == 0)[0]) == 0:
            num_nonaircraft += 1
            continue

        # Compute individual metrics and scores
        ind_metrics = compute_individual_metrics(
            sequences=scene['sequences'].copy(), hold_lines=hl_xy.copy())

        ind_scores, ind_scene_score = S.compute_individual_scores(
            metrics=ind_metrics, agent_types=agent_types)

        # Compute interaction metrics and scores
        int_metrics = compute_interaction_metrics(
            sequences=scene['sequences'].copy(), agent_types=agent_types, hold_lines=hl_xy.copy(),
            graph_map=graph_map, agent_to_agent_dist_thresh=rwy_ext)
        if np.where(int_metrics['status'] == s.OK)[0].shape[0] == 0:
            num_noninteractive += 1
            continue

        # Compute individual and interaction scores
        int_metrics['num_agents'] = agent_types.shape[0]
        int_scores, int_scene_score = S.compute_interaction_scores(metrics=int_metrics)

        if args.plot:
            # filetag = os.path.join(outdir, f"{subdir}_{scenario_id}_s-{round(int_scene_score, 3)}_int")
            # C.plot_scene_scores(scene, assets, int_scores, int_scene_score, filetag)

            # filetag = os.path.join(outdir, f"{subdir}_{scenario_id}_s-{round(ind_scene_score, 3)}_ind")
            # C.plot_scene_scores(scene, assets, ind_scores, ind_scene_score, filetag)

            scores = ind_scores + int_scores
            scene_score = ind_scene_score + int_scene_score
            filetag = os.path.join(outdir, f"{subdir}_{scenario_id}_s-{round(scene_score, 3)}")
            C.plot_scene_scores(scene, assets, scores, scene_score, filetag, show_highest=True)

            # outdir = os.path.join(args.out_dir, "vis", args.airport)
            # os.makedirs(outdir, exist_ok=True)
            # filetag = os.path.join(outdir, f"{subdir}_{scenario_id}")
            # C.plot_scene(scene, assets, filetag)

    print(f"Processed scenarios: {num_scenes}")
    print(f"\tNon Aircraft scenarios: {num_nonaircraft} ({round(num_nonaircraft/num_scenes, 3)})")
    print(
        f"\tNon Interactive scenarios: {num_noninteractive} ({round(num_noninteractive/num_scenes, 3)})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", type=str, default="ksea", choices=[SUPPORTED_AIRPORTS])
    parser.add_argument("--base_dir", type=str, default="../datasets/swim")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--num_scenes", type=int, default=-1)
    args = parser.parse_args()

    run(**vars(args))