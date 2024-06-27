import os

from easydict import EasyDict

from amelia_scenes.utils.common import SUPPORTED_AIRPORTS

def run(
    airport: str, 
    base_dir: str, 
    traj_version: str, 
    graph_version: str, 
    parallel: bool, 
    overwrite: bool, 
    perc_process: float,
    to_process: str,
    seed: int
) -> None:
    traj_data_dir = f"traj_data_{traj_version}"
    if to_process == 'scenes':
        from amelia_scenes.scene_processing.src.scene_processor import SceneProcessor as Pr
        in_data_dir = os.path.join(base_dir, traj_data_dir, 'raw_trajectories')
        out_data_dir = os.path.join(base_dir, traj_data_dir, 'proc_scenes')
    elif to_process == 'metas':
        from amelia_scenes.scene_processing.src.scene_meta_processor import SceneMetaProcessor as Pr
        in_data_dir = os.path.join(base_dir, traj_data_dir, 'proc_scenes')
        out_data_dir = os.path.join(base_dir, traj_data_dir, 'proc_scenes_meta')
    else:
        from amelia_scenes.scene_processing.src.full_processor import SceneProcessor as Pr
        in_data_dir = os.path.join(base_dir, traj_data_dir, 'raw_trajectories')
        out_data_dir = os.path.join(base_dir, traj_data_dir, 'proc_full_scenes')
    
    # TODO: provide configs as YAML files
    config = EasyDict({
        "airport": airport,
        "in_data_dir": in_data_dir,
        "out_data_dir": out_data_dir,   
        'graph_data_dir': os.path.join(base_dir,  f"graph_data_{graph_version}"),
        "assets_dir": os.path.join(base_dir, 'assets'),
        "parallel": parallel,   
        "perc_process": perc_process,
        'overwrite': overwrite,
        "pred_lens": [20, 50],
        "hist_len": 10,
        "skip": 1,
        "min_agents": 2, 
        "max_agents": 15,
        "min_valid_points": 2,
        "seed": seed
    })

    Pr(config=config).process_data()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", type=str, default="kbos", choices=SUPPORTED_AIRPORTS)
    parser.add_argument("--base_dir", type=str, default="../../datasets/amelia")
    parser.add_argument("--traj_version", type=str, default="a10v07")
    parser.add_argument("--graph_version", type=str, default="a10v01os")
    parser.add_argument("--parallel", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--perc_process", type=float, default=1.0)
    parser.add_argument("--to_process", default='both', choices=['scenes', 'metas', 'both'])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(**vars(args))