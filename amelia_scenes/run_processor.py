import os
import hydra
import copy
import amelia_scenes.utils.common as C

from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf, open_dict

def _build_paths(cfg: DictConfig) -> None:
    """Derive all path fields in-place from high-level flags."""
    tag = "benchmark" if cfg.benchmark else "xplane" if cfg.xplane else cfg.traj_version
    traj_dir = f"traj_data_{tag}"
    with open_dict(cfg):
        cfg.in_data_dir  = os.path.join(cfg.base_dir, traj_dir,
                                        "raw_trajectories" if cfg.to_process in ["scenes", "full"]
                                        else "proc_scenes")
        cfg.out_data_dir = os.path.join(cfg.base_dir, traj_dir,
                                        {"scenes": f"proc_scenes_{cfg.out_version}",
                                        "metas" : f"proc_scenes_{cfg.out_version}",
                                        "full"  : f"proc_scenes_{cfg.out_version}"}[cfg.to_process])
        cfg.bench_data_dir = os.path.join(cfg.base_dir, traj_dir, "benchmark")
        cfg.graph_data_dir = os.path.join(cfg.base_dir, f"graph_data_{cfg.graph_version}")
        cfg.assets_dir     = os.path.join(cfg.base_dir, "assets")
    
    # Make sure the output directory exists
    # os.makedirs(cfg.out_data_dir, exist_ok=True)

@hydra.main(config_path="conf", config_name="processor_conf", version_base=None)
def run(cfg: DictConfig) -> None:
    # Load airport
    supported_airports = C.get_available_airports(cfg.base_dir)
    airports = supported_airports if cfg.airport == "all" else [cfg.airport]

    # Sanity checks
    assert not (cfg.benchmark and cfg.xplane), "[ ERROR ]Cannot use both benchmark and xplane!"
    assert len(airports) > 0, "[ ERROR ] No airports found, check paths!"
    assert cfg.to_process in ["scenes", "metas", "full"], "[ ERROR ] Invalid processing mode!"
    assert set(airports).issubset(supported_airports), f"[ ERROR ] Airport {cfg.airport} not supported!"
    
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
        # try:
        #     processor_cls(config=cfg_one).process_data()
        # except Exception as e:
        #     print(f"Error processing {airport}: {e}")

if __name__ == "__main__":
    run()
