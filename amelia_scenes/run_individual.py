import scene_utils.common as C
import scene_utils.scores as S
import scene_utils.individual_metrics as M
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
random.seed(42)
np.set_printoptions(suppress=True)


SUBDIR = __file__.split('/')[-1].split('.')[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="../datasets/swim")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--airport", type=str, default="ksea",
                        choices=["ksea", "kbos", "kmdw", "kewr"])
    parser.add_argument("--split", type=str, default='train_month')
    parser.add_argument("--num_scenes", type=int, default=-1)
    parser.add_argument("--perc", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--compute_scores", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.base_dir, "raw_trajectories", f"{args.split}.txt"), "r") as f:
        subdirs = f.read().splitlines()

    scenarios_subdirs = [os.path.join(
        args.base_dir, "proc_trajectories", f) for f in subdirs]

    airport = args.airport
    print(f"Processing airport: {airport.upper()}")

    map_dir = os.path.join(args.base_dir, "maps", airport)
    assets = C.load_assets(map_dir=map_dir)

    scenarios = []
    for scenario_subdir in scenarios_subdirs:
        if airport not in scenario_subdir:
            continue
        scenarios_list = glob.glob(
            f"{scenario_subdir.removesuffix('.csv')}/*.pkl", recursive=True)
        scenarios += scenarios_list
    num_scenarios = len(scenarios)

    perc = int(args.perc * 100)
    singles_filepath = f'./out/run_cluster_anomaly/{airport}_{perc}/{args.split}_singles_anomaly.npz'
    singles_anomaly = np.load(singles_filepath, allow_pickle=True)[
        'arr_0'].item()

    scenarios = list(set(list(singles_anomaly.keys())).intersection(scenarios))
    random.shuffle(scenarios)
    if args.num_scenes > 0:
        scenarios = scenarios[:args.num_scenes]
    print(f"Processing {perc * 100}% of the scenarios: {len(scenarios)}")

    # Hold line locations
    raster_map, hold_lines, graph_map, ll_extent, agents = assets
    hl_xy = hold_lines[:, 2:4]
    hl_ll = hold_lines[:, :2]

    combined_metrics = {
        'waiting_period_L': [],
        'waiting_period_C': [],
        'acceleration_rate': [],
        'deceleration_rate': [],
        'speed': [],
        'jerk': [],
        'traj_anomaly': []
    }

    combined_scores = []
    combined_scene_scores = []

    scene_scores_pickle = {}
    agents_scores_pickle = {}
    for n, scenario in enumerate(tqdm(scenarios)):
        split = scenario.split("/")
        subdir, scenario_id = split[-2], split[-1].split('.')[0]

        with open(scenario, 'rb') as f:
            scene = pickle.load(f)

        agent_types = np.asarray(scene['agent_types'])
        if len(np.where(agent_types == 0)[0]) == 0:
            continue

        metrics = M.compute_individual_metrics(
            sequences=scene['sequences'], hold_lines=hl_xy)
        metrics['traj_anomaly'] = singles_anomaly[scenario]
        for k, v in metrics.items():
            if 'waiting' in k:
                v = [t for t, d, i in v]
            elif 'rate' in k:
                v = [a for a, m in v]
            combined_metrics[k] += v

        if args.compute_scores:
            metrics['num_agents'] = agent_types.shape[0]
            scores, scene_score = S.compute_individual_scores(
                metrics, agent_types)
            combined_scores += scores.tolist()
            combined_scene_scores.append(scene_score)

            scene_scores_pickle[scenario] = scene_score
            agents_scores_pickle[scenario] = scores

            if args.plot:
                outdir = os.path.join(out_dir, args.airport, "scene_scores")
                os.makedirs(outdir, exist_ok=True)
                filetag = os.path.join(
                    outdir, f"{subdir}_{scenario_id}_score-{round(scene_score, 2)}")
                C.plot_scene_scores(scene, assets, scores,
                                    scene_score, filetag, show_highest=True)

    outdir = os.path.join(out_dir, f"{airport}_{perc}", "metric_distribution")
    os.makedirs(outdir, exist_ok=True)
    for k, v in combined_metrics.items():
        print(f"{k}")
        v = np.asarray(v)
        plt.hist(v, bins='auto', density=True)
        plt.title(
            f"Airport: {airport.upper()} Metric: {k} Split: {args.split}")
        plt.savefig(f"{outdir}/{args.split}_{k}.png", dpi=600)
        plt.close()

    if args.compute_scores:
        combined_scores = np.asarray(combined_scores)
        plt.hist(combined_scores, bins='auto', density=True)
        plt.title(f"Airport: {airport.upper()} Split: {args.split}")
        plt.savefig(f"{outdir}/{args.split}_individual_scores.png", dpi=600)
        plt.close()

        combined_scene_scores = np.asarray(combined_scene_scores)
        plt.hist(combined_scene_scores, bins='auto', density=True)
        plt.savefig(f"{outdir}/{args.split}_scene_scores.png", dpi=600)
        plt.close()

    with open(os.path.join(out_dir, f"{airport}_{perc}", f'{args.split}_scenarios_scores.pkl'), 'wb') as pkl:
        pickle.dump(scene_scores_pickle, pkl)

    with open(os.path.join(out_dir, f"{airport}_{perc}", f'{args.split}_agents_scores.pkl'), 'wb') as pkl:
        pickle.dump(agents_scores_pickle, pkl)