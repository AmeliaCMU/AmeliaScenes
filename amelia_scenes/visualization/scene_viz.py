import glob
import matplotlib.pyplot as plt 
import numpy as np
import os

from matplotlib import cm
from matplotlib.offsetbox import AnnotationBbox
from natsort import natsorted
from PIL import Image
from typing import Tuple

from amelia_scenes.visualization import common as C
from amelia_scenes.visualization import benchmark_viz as bench
from amelia_scenes.visualization import scoring_viz as scoring
from amelia_scenes.utils import global_masks as G


SUPPORTED_SCENES_TYPES = ['simple', 'benchmark', 'benchmark_pred', 'features', 'scores', 'strategy']

def plot_scene(
    scenario: dict, 
    assets: Tuple, 
    filetag: str, 
    scene_type: str = 'simple',
    predictions: Tuple = None,
    scores: bool = False,
    features_to_add: list = [],
    features: dict = {},
    dpi = 200
) -> None:
    """ Plots a TrajAir scene. If features are specified, it'll create subplots, where each plot
    shows the agent's colored by metric value.
    Inputs
    ------
        scenario[dict]: dictionary containing the scene components. 
        assets[Tuple]: tuple containing all scene assets (e.g., map, graph, agent assets, etc.)
        filetag[str]: nametag for the image to save.
        features_to_add[list]: list of features to create subplots for.
        features[dict]: computed features for each agent.
        dpi[int]: image dpi to save.
    """
    reproject = True if scenario['airport_id'] in ['panc', 'kmsy'] else False
    agent_sequences = scenario['agent_sequences'][:, :, G.HLL]
    agent_masks = scenario['agent_masks']
    agent_types = scenario['agent_types']
    agent_ids = scenario['agent_ids']
    
    agents = (agent_sequences, agent_masks, agent_types, agent_ids)

    assert scene_type in SUPPORTED_SCENES_TYPES, f"Scene type not supported {scene_type}"
    
    if scene_type == 'simple':
        plot_scene_simple(agents, assets, tag=filetag, dpi=dpi, reproject=reproject)
    elif scene_type == 'benchmark':
        benchmark = scenario['benchmark']
        bench.plot_scene_benchmark(
            agents, assets, benchmark, tag=filetag, dpi=dpi, reproject=reproject)
    elif scene_type == 'benchmark_pred':
        benchmark = scenario['benchmark']
        bench.plot_scene_benchmark_preds(
            agents, assets, benchmark, predictions, tag=filetag, dpi=dpi, reproject=reproject, 
            hist_len=scenario['hist_len'], ego_agent_ids=scenario['ego_agent_ids'])
    elif scene_type == 'strategy':
        raise NotImplementedError
        agent_order = scenario['meta']['agent_order']
        scoring.plot_scene_strategy(
            agent_sequences, agent_order, assets, agent_masks, agent_types, agent_ids, tag=filetag, 
            dpi=dpi)
    elif scene_type == 'scores':
        raise NotImplementedError
        agents_interest = benchmark['bench_agents']
        agent_scores = scenario['meta']['agent_scores'] if scores else None
        scoring.plot_scene_scores(
            agent_sequences, agent_scores, assets, agent_masks, agent_types, agent_ids, tag=filetag, 
            agents_interest=agents_interest, dpi=dpi)
    else:
        raise NotImplementedError
        scoring.plot_scene_features(scenario, assets, filetag, features_to_add, features, dpi=dpi)

def plot_scene_simple(
    agents: Tuple, assets: Tuple, agents_interest: list = [], tag: str = 'temp.png', dpi=600, 
    reproject: bool = False, projection: str = 'EPSG:3857'
) -> None:
    """ Visualize simple scenes """
    agent_sequences, agent_masks, agent_types, agent_ids = agents
    bkg, hold_lines, graph_nx, limits, agents = assets
    north, east, south, west, z_min, z_max = limits
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=.2) 

    # Display each trajectory
    traj_color, traj_lw = C.MOTION_COLORS['gt_hist'][0], C.MOTION_COLORS['gt_hist'][1]
    zipped = zip(agent_sequences, agent_types, agent_masks, agent_ids)
    for n, (trajectory, agent_type, mask, agent_id) in enumerate(zipped):
        traj = trajectory[mask]
        if traj.shape[0] == 0:
            continue
        # Get heading at last point of trajectory history.
        heading = traj[-1, 0] 
        traj_ll = C.reproject_sequences(traj[:, 1:], projection) if reproject else traj[:, 1:]
        lon, lat = traj_ll[-1, 1], traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type= int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, heading, zoom=C.ZOOM[agent_type])
        if agent_id in agents_interest:
            ax.scatter(lon, lat, color='#F29C3A', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        ax.add_artist(ab)
        ax.plot(traj_ll[:, 1], traj_ll[:, 0], color=traj_color, lw=traj_lw) 
    
    # Plot movement
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # if lat < abs(north) and lat > south and lon > west and lon < east:
        # ax.axis([np.min(all_lon) - C.OFFSET, np.max(all_lon) + C.OFFSET, np.min(all_lat) - C.OFFSET, np.max(all_lat) + C.OFFSET])
    # axis(xmin, xmax, ymin, ymax)
    # ax.axis([west - C.OFFSET, east + C.OFFSET, south - C.OFFSET, north + C.OFFSET])

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

# def plot_scene_animation(
#     asset_dir: str, 
#     scene: dict, 
#     geodesic: Geodesic, 
#     tag: str, 
#     out_dir: str = './out', 
#     dim: int = 2, 
#     hist_len: int = 10,
#     plot_full_scene = False,
#     k_agents:int = 5,
#     plot_n: int = 20
# ) -> None:

#     # TODO: pre-load these in trajpred.py
#     rasters, ref_ll, extent_ll = {}, {}, {}
#     agents = {
#         C.AIRCRAFT: imageio.imread(os.path.join(asset_dir, 'ac.png')),
#         C.VEHICLE: imageio.imread(os.path.join(asset_dir, 'vc.png')),
#         C.UNKNOWN: imageio.imread(os.path.join(asset_dir, 'uk_ac.png'))
#     } 
    
#     # TODO: update actual asset 
#     image = agents[C.UNKNOWN]
#     image = image.astype(np.float32)
#     image = image * 1.35
#     image = np.clip(image, 0, 255)
#     image = image.astype(np.uint8)    
#     agents[C.UNKNOWN] = image
    
#     num_agents = scene['num_agents']
        
#     sequences = scene['sequences']   # B, N, T, D=9
#     num_agents = scene['num_agents']   # B
#     agent_types = scene['agent_types']
#     airport_id = scene['airport_id']   # B
#     scenario_id = scene['scenario_id']   # B
    
#     # TODO: preload these assets in trajpred.py
#     if rasters.get(airport_id) is None:
#         im = cv2.imread(os.path.join(asset_dir, airport_id, 'bkg_map.png'))
#         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         im = cv2.resize(im, (im.shape[0]//2, im.shape[1]//2))
#         rasters[airport_id] = im
        
#         with open(os.path.join(asset_dir, airport_id,'limits.json'), 'r') as fp:
#             ref_data = EasyDict(json.load(fp))
#         ref_ll[airport_id] = [ref_data.ref_lat, ref_data.ref_lon, ref_data.range_scale]
#         espg = ref_data.espg_4326
#         extent_ll[airport_id] = (espg.north, espg.east, espg.south, espg.west)

#         maps = (rasters[airport_id].copy(), agents)
        
#         # Read sequences from batch
#         gt_abs_traj = sequences[:num_agents] # N, T, D
#         N, T, D = gt_abs_traj.shape

#         # Transform relative XY prediction to absolute LL space
#         tag_i = f"{airport_id}_scene-{scenario_id}_{tag}"
        
#         for t in range(1, T):
#             temp_tag = f'temp_{t}'
#             plot_scene(
#                 trajectories = gt_abs_traj[:, :t, G.HLL], 
#                 maps = maps, 
#                 ll_extent = extent_ll[airport_id], 
#                 tag = temp_tag,  
#                 out_dir = out_dir,
#                 agent_types=agent_types
#             )
        
#         files = natsorted(glob.glob(f"{out_dir}/temp*.png"))
#         imgs = [Image.open(f) for f in natsorted(glob.glob(f"{out_dir}/temp*.png"))]
#         imgs[0].save(
#             f'{out_dir}/{tag_i}.gif', format='GIF', append_images=imgs[1:], save_all=True,
#             duration=200, loop=0)
#         for f in files:
#             os.remove(f)


