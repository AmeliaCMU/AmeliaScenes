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
from amelia_scenes.utils import global_masks as G

def plot_scene(
    scenario: dict, 
    assets: Tuple, 
    filetag: str, 
    scene_type: str = 'simple',
    scores: bool = False,
    features_to_add: list = [],
    features: dict = {},
    dpi = 600
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
    agent_sequences = scenario['agent_sequences'][:, :, G.HLL]
    agent_masks = scenario['agent_masks']
    agent_types = scenario['agent_types']
    agent_ids = scenario['agent_ids']

    agents = (agent_sequences, agent_masks, agent_types, agent_ids)

    assert scene_type in ['simple', 'benchmark', 'features', 'scores', 'strategy'], \
        f"Scene type not supported {scene_type}"
    
    if scene_type == 'simple':
        plot_scene_simple(agents, assets, tag=filetag, dpi=dpi)
    elif scene_type == 'benchmark':
        benchmark = scenario['benchmark']
        plot_scene_benchmark(agents, assets, tag=filetag, benchmark=benchmark, dpi=dpi)
    elif scene_type == 'strategy':
        raise NotImplementedError
        agent_order = scenario['meta']['agent_order']
        plot_scene_strategy(
            agent_sequences, agent_order, assets, agent_masks, agent_types, agent_ids, tag=filetag, 
            dpi=dpi)
    elif scene_type == 'scores':
        raise NotImplementedError
        agents_interest = benchmark['bench_agents']
        agent_scores = scenario['meta']['agent_scores'] if scores else None
        plot_scene_scores(
            agent_sequences, agent_scores, assets, agent_masks, agent_types, agent_ids, tag=filetag, 
            agents_interest=agents_interest, dpi=dpi)
    else:
        raise NotImplementedError
        plot_scene_features(scenario, assets, filetag, features_to_add, features, dpi=dpi)

def plot_scene_simple(
    agents: Tuple, assets: Tuple, agents_interest: list = [], tag: str = 'temp.png', dpi=600
) -> None:
    """ Visualize simple scenes """
    agent_sequences, agent_masks, agent_types, agent_ids = agents
    bkg, hold_lines, graph_nx, limits, agents = assets
    north, east, south, west, z_min, z_max = limits

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=.2) 

    zipped = zip(agent_sequences, agent_types, agent_masks, agent_ids)
    for n, (trajectory, agent_type, mask, agent_id) in enumerate(zipped):
    
        trajectory = trajectory[mask]
        if trajectory.shape[0] == 0:
            continue
        # Get heading at last point of trajectory history.
        gt_traj_ll, gt_heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = gt_traj_ll[-1, 1], gt_traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type= int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
        if agent_id in agents_interest:
            ax.scatter(lon, lat, color='#F29C3A', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        ax.add_artist(ab)
        # Plot past sequence
        ax.plot(
            gt_traj_ll[:, 1], gt_traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
            lw = C.MOTION_COLORS['gt_hist'][1]) 
    
    # Plot movement
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_scene_benchmark(
    agents: Tuple, assets: Tuple, benchmark: dict = None, tag: str = 'temp.png', dpi=600
) -> None:
    """ Visualizing benchmark scenes. Like 'plot_scene_simple' but adds benchmark metadata """
    agent_sequences, agent_masks, agent_types, agent_ids = agents    
    bkg, hold_lines, graph_nx, limits, agents = assets
    north, east, south, west, z_min, z_max = limits

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.5) 
    
    agents_interest = benchmark['bench_agents']
    halo_values = benchmark['timestep']
    airport_name = benchmark['airport'].values[0]
    date = benchmark['date'].values[0]
    ax.set_title(f"{airport_name.upper()} ({date})")

    zipped = zip(agent_sequences, agent_types, agent_masks, agent_ids)
    for n, (trajectory, agent_type, mask, agent_id) in enumerate(zipped):
    
        trajectory = trajectory[mask]
        if trajectory.shape[0] == 0:
            continue
        # Get heading at last point of trajectory history.
        gt_traj_ll, gt_heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = gt_traj_ll[-1, 1], gt_traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type= int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
        if agent_id in agents_interest:
            color = cm.autumn(halo_values) # '#F29C3A'
            ax.scatter(lon, lat, color=color, alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        ax.add_artist(ab)
        # Plot past sequence
        ax.plot(
            gt_traj_ll[:, 1], gt_traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
            lw = C.MOTION_COLORS['gt_hist'][1]) 
    
    # Plot movement
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_scene_strategy(
    agent_sequences: np.array, 
    agent_order: dict,
    assets: Tuple, 
    agent_masks: np.array = None, 
    agent_types: np.array = None, 
    agent_ids: np.array = None,
    k_agents: int = 6,
    tag: str = 'temp.png', 
    dpi=600
) -> None:
    """ Tool for visualizing marginal model predictions for the ego-agent. """
    mm =  C.MOTION_COLORS['multi_modal']
    
    bkg, hold_lines, graph_nx, limits, agents = assets
    north, east, south, west, z_min, z_max = limits

    fig, axs = plt.subplots(1, 2, figsize=(30, 80))
    
    random_order = agent_order['random'][:k_agents]
    critical_order = agent_order['critical'][:k_agents]

    # Display global map
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.8) 

    zipped = zip(agent_sequences, agent_types, agent_masks, agent_ids)
    for n, (trajectory, agent_type, mask, agent_id) in enumerate(zipped):
        trajectory = trajectory[mask]
        if trajectory.shape[0] == 0:
            continue

        # Plot scene only
        # Get heading at last point of trajectory history.
        gt_traj_ll, gt_heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = gt_traj_ll[-1, 1], gt_traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type = int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
        
        if n in random_order:
            axs[0].scatter(lon, lat, color='#F39C12', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)

        if n in critical_order:
            axs[1].scatter(lon, lat, color='#F39C12', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        axs[0].add_artist(ab)

        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        axs[1].add_artist(ab)
    
        axs[0].plot(
            gt_traj_ll[:, 1], gt_traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
            lw = C.MOTION_COLORS['gt_hist'][1]) 
        axs[1].plot(
            gt_traj_ll[:, 1], gt_traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
            lw = C.MOTION_COLORS['gt_hist'][1])     
            
    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(hspace=0.1) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_scene_scores(
    agent_sequences: np.array, 
    agent_scores: dict,
    assets: Tuple, 
    agent_masks: np.array = None, 
    agent_types: np.array = None, 
    agent_ids: np.array = None,
    agents_interest: list = [],
    k_agents: int = 5,
    tag: str = 'temp.png', 
    dpi=600
) -> None:
    """ Tool for visualizing marginal model predictions for the ego-agent. """
    mm =  C.MOTION_COLORS['multi_modal']
    
    bkg, hold_lines, graph_nx, limits, agents = assets
    north, east, south, west, z_min, z_max = limits

    fig, axs = plt.subplots(1, 1+len(agent_scores.keys()), figsize=(30, 80))

    k_agents_ids = {}
    for score_type, score_values in agent_scores.items():
        norm_scores = C.norm(score_values)
        agent_scores[score_type] = norm_scores
        k_agents_ids[score_type] = np.argsort(norm_scores)[::-1][:k_agents]

    # Display global map
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.4) 

    zipped = zip(agent_sequences, agent_types, agent_masks, agent_ids)
    for n, (trajectory, agent_type, mask, agent_id) in enumerate(zipped):
        trajectory = trajectory[mask]
        if trajectory.shape[0] == 0:
            continue

        # Plot scene only
        axs[0].set_title('Scene')
        # Get heading at last point of trajectory history.
        gt_traj_ll, gt_heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = gt_traj_ll[-1, 1], gt_traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type = int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
        if agent_id in agents_interest:
            axs[0].scatter(lon, lat, color='#66B7F7', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        axs[0].add_artist(ab)
        # Plot past sequence
        axs[0].plot(
            gt_traj_ll[:, 1], gt_traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
            lw = C.MOTION_COLORS['gt_hist'][1]) 

        # Plot scored
        for i, (score_type, score_values) in enumerate(agent_scores.items()):
            if n in k_agents_ids[score_type]:
                axs[i+1].scatter(
                    lon, lat, color='#F29C3A', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
            axs[i+1].set_title(score_type.capitalize())
            score = cm.autumn(1.0 - score_values[n])
            ab = AnnotationBbox(img, (lon, lat), frameon = False) 
            axs[i+1].add_artist(ab)
            axs[i+1].scatter(
                gt_traj_ll[:, 1], gt_traj_ll[:, 0], color=score, s = C.MOTION_COLORS['gt_hist'][1])
            axs[i+1].text(lon, lat, s=round(score_values[n], 2), color='black', fontsize='x-small')
            
    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_scene_features():
    raise NotImplemented

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

def to_gif(base_dir, scene_tag):
    files = natsorted(glob.glob(f"{base_dir}/{scene_tag}*.png"))
    imgs = [Image.open(f) for f in natsorted(glob.glob(f"{base_dir}/{scene_tag}*.png"))]
    imgs[0].save(
        f'{base_dir}/{scene_tag}.gif', format='GIF', append_images=imgs[1:], save_all=True,
        duration=200, loop=0)
    
    for f in files:
        os.remove(f)