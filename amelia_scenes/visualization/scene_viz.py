# import cv2
import numpy as np
# import imageio.v2 as imageio
# import json 
# import glob
# import os
# import shutil
# import random
# import torch

# import amelia_viz.common as C
# import amelia_viz.marginal_predictions as M
import matplotlib.pyplot as plt 

# from easydict import EasyDict
# from torch import tensor
from matplotlib.offsetbox import AnnotationBbox
from typing import Tuple
# from geographiclib.geodesic import Geodesic

from amelia_scenes.visualization import common as C
from amelia_scenes.utils import global_masks as G

# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
# import numpy as np

# from typing import Tuple

# from natsort import natsorted

# import amelia_viz.common as C

# from PIL import Image

def plot_scene(
    scenario: dict, 
    assets: Tuple, 
    filetag: str, 
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
    agent_types = scenario['agent_types']
    agent_sequences = scenario['agent_sequences'][:, :, G.HLL]
    agent_masks = scenario['agent_masks']
    if not len(features_to_add):
        plot_scene_simple(agent_sequences, assets, agent_masks, agent_types, filetag, dpi=dpi)
    else:
        raise NotImplementedError
        plot_scene_features(scenario, assets, filetag, features_to_add, features, dpi=dpi)

def plot_scene_simple(
    sequences: np.array, 
    assets: Tuple, 
    masks: np.array = None, 
    agent_types: np.array = None, 
    tag: str = 'temp.png', 
    ego_id: int = 0, 
    agents_interest: list = [],
    dpi=600
) -> None:
    """ Tool for visualizing marginal model predictions for the ego-agent. """
    mm =  C.MOTION_COLORS['multi_modal']
    
    bkg, hold_lines, graph_nx, limits, agents = assets
    north, east, south, west, z_min, z_max = limits
    
    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=.2) 
    # ax.set_facecolor('slategrey')
    
    # Iterate through agent in the scene
    # all_lon, all_lat, scores = [], [], []

    for n, (trajectory, agent_type, mask) in enumerate(zip(sequences, agent_types, masks)):
    
        trajectory = trajectory[mask]
        # Get heading at last point of trajectory history.
        gt_traj_ll, gt_heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = gt_traj_ll[-1, 1], gt_traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type= int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, gt_heading, zoom=C.ZOOM[agent_type])
        if n in agents_interest:
            if n != ego_id:
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