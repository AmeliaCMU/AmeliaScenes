import os
import hydra
import copy
import amelia_scenes.utils.common as C

from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig

def _build_paths(cfg: DictConfig) -> None:
    """Derive all path fields in-place from high-level flags."""
    tag = "benchmark" if cfg.benchmark else "xplane" if cfg.xplane else cfg.traj_version
    traj_dir = f"traj_data_{tag}"
    # Add paths to config dictionary
    with open_dict(cfg):
        # Add  source data directories
        cfg.in_data_dir  = os.path.join(cfg.base_dir, traj_dir,
                                        "raw_trajectories" if cfg.to_process in ["scenes", "full"]
                                        else "proc_scenes")
        # Make output directories
        out_dir = os.path.join(cfg.base_dir, traj_dir,
                                        {"scenes": f"proc_scenes",
                                        "metas" : f"proc_scenes_metas",
                                        "full"  : f"proc_full_scenes"}[cfg.to_process])
        if cfg.out_version is not None:
            out_dir = f"{out_dir}_{cfg.out_version}"
        cfg.out_data_dir = out_dir
        # Add assets directory
        cfg.bench_data_dir = os.path.join(cfg.base_dir, traj_dir, "benchmark")
        cfg.graph_data_dir = os.path.join(cfg.base_dir, f"graph_data_{cfg.graph_version}")
        cfg.assets_dir     = os.path.join(cfg.base_dir, "assets")
    
    # Make sure the output directory exists
    os.makedirs(cfg.out_data_dir, exist_ok=True)
    
def _parse_airport(airport, supported_airports=None) -> list:
    if isinstance(airport, str):
        airports = supported_airports if airport == "all" else [airport]
    elif not isinstance(airport, ListConfig):
        raise ValueError("Airport must be a string or a list of strings.")
    airports = list(airport)
    assert set(airports).issubset(supported_airports), f"[ ERROR ] Airport {cfg.airport} not supported!"
    return airports

@hydra.main(config_path="conf", config_name="processor_conf", version_base=None)
def run(cfg: DictConfig) -> None:
    # Load airport
    supported_airports = C.get_available_airports(cfg.base_dir)
    airports = _parse_airport(cfg.airport, supported_airports)
 
    # Sanity checks
    assert not (cfg.benchmark and cfg.xplane), "[ ERROR ]Cannot use both benchmark and xplane!"
    assert len(airports) > 0, "[ ERROR ] No airports found, check paths!"
    assert cfg.to_process in ["scenes", "metas", "full"], "[ ERROR ] Invalid processing mode!"
    
    # Run processor for each airport
    for airport in airports:
        cfg_one = copy.deepcopy(cfg)   # Modify config for compatibility with each airport.
        cfg_one.airport = airport
        _build_paths(cfg_one)
        print(f"\033[93m[INFO]\033[0m Processing airport: {cfg_one.airport}")
        for key, value in cfg_one.items():
            print(f"\033[94m{key}\033[0m: \033[97m{value}\033[0m")
            
        # Intantiate processor     
        processor_cls = hydra.utils.get_class(cfg_one._target_)
        try:
            processor_cls(config=cfg_one).process_data()
        except Exception as e:
            print(f"Error processing {airport}: {e}")

if __name__ == "__main__":
    run()
