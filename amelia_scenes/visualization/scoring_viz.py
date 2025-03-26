import matplotlib.pyplot as plt 
import numpy as np

from matplotlib import cm
from matplotlib.offsetbox import AnnotationBbox
from typing import Tuple

from amelia_scenes.visualization import common as C

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
        traj_ll, heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = traj_ll[-1, 1], traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type = int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, heading, zoom=C.ZOOM[agent_type])
        
        if n in random_order:
            axs[0].scatter(lon, lat, color='#F39C12', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)

        if n in critical_order:
            axs[1].scatter(lon, lat, color='#F39C12', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        axs[0].add_artist(ab)

        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        axs[1].add_artist(ab)
    
        axs[0].plot(
            traj_ll[:, 1], traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
            lw = C.MOTION_COLORS['gt_hist'][1]) 
        axs[1].plot(
            traj_ll[:, 1], traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
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
    
    bkg, hold_lines, graph_nx, (limits, ref_data), agents = assets
    north, east, south, west, z_min, z_max = limits

    fig, axs = plt.subplots(1, 1+len(agent_scores.keys()), figsize=(30, 80))

    k_agents_ids = {}
    for score_type, score_values in agent_scores.items():
        # breakpoint()
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
        traj_ll, heading = trajectory[:, 1:], trajectory[-1, 0] 
        lon, lat = traj_ll[-1, 1], traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue

        agent_type = int(agent_type)
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = C.plot_agent(icon, heading, zoom=C.ZOOM[agent_type])
        if agent_id in agents_interest:
            axs[0].scatter(lon, lat, color='#66B7F7', alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)
                
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        axs[0].add_artist(ab)
        # Plot past sequence
        axs[0].plot(
            traj_ll[:, 1], traj_ll[:, 0], color = C.MOTION_COLORS['gt_hist'][0], 
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
                traj_ll[:, 1], traj_ll[:, 0], color=score, s = C.MOTION_COLORS['gt_hist'][1])
            axs[i+1].text(lon, lat, s=round(score_values[n], 2), color='black', fontsize='x-small')
            
    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_scene_features():
    raise NotImplemented