import itertools
import matplotlib.pyplot as plt 
import numpy as np
import os
import torch 

from geographiclib.geodesic import Geodesic
from matplotlib import cm
from matplotlib.offsetbox import AnnotationBbox
from natsort import natsorted
from PIL import Image
from typing import Tuple

from amelia_scenes.visualization import common as C
from amelia_scenes.utils import global_masks as G
from amelia_scenes.scoring.interactive import compute_collisions
from src.utils.transform_utils import xy_to_ll

SUPPORTED_SCENES_TYPES = ['simple', 'benchmark', 'benchmark_pred', 'features', 'scores', 'strategy']

def plot_scene_benchmark(
    agents: Tuple, 
    assets: Tuple, 
    benchmark: dict, 
    tag: str = 'temp.png', 
    dpi=600, 
    reproject: bool = False, 
    projection: str = 'EPSG:3857'
) -> None:
    """ Visualizing benchmark scenes. Like 'plot_scene_simple' but adds benchmark metadata """
    agent_sequences, agent_masks, agent_types, agent_ids = agents    
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.5) 
    
    agents_interest = benchmark['bench_agents']
    halo_values = benchmark['timestep']
    airport_name = benchmark['airport'].values[0]
    date = benchmark['date'].values[0]
    ax.set_title(f"{airport_name.upper()} ({date})")

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

        # Plot agent icon on last point of sequence. If it's a benchmark agent add halo.
        agent_type = int(agent_type)
        icon = agents[agent_type]
        img = C.plot_agent(icon, heading, zoom=C.ZOOM[agent_type])
        if agent_id in agents_interest:
            color = cm.autumn(halo_values) # '#F29C3A'
            ax.scatter(lon, lat, color=color, alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)        
        ab = AnnotationBbox(img, (lon, lat), frameon = False) 
        ax.add_artist(ab)

        # Plot rest of the sequence
        ax.plot(traj_ll[:, 1], traj_ll[:, 0], color=traj_color, lw=traj_lw) 
    
    # Plot movement
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axis([west - C.OFFSET, east + C.OFFSET, south - C.OFFSET, north + C.OFFSET])

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_scene_benchmark_preds(
    agents: Tuple, 
    assets: Tuple, 
    benchmark: dict, 
    predictions: Tuple,
    tag: str = 'temp.png', 
    dpi=600, 
    reproject: bool = False, 
    projection: str = 'EPSG:3857',
    hist_len: int = 10,
    ego_agent_ids: list = None
) -> None:
    """ Visualizing benchmark scenes. Like 'plot_scene_simple' but adds benchmark metadata """
    agent_sequences, agent_masks, agent_types, agent_ids = agents    
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    geodesic = Geodesic.WGS84
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.5) 
    
    agents_interest = benchmark['bench_agents']
    halo_values = benchmark['timestep']
    airport_name = benchmark['airport'].values[0]
    date = benchmark['date'].values[0]
    plt.title(f"{airport_name.upper()} ({date})")

    hist_color, hist_lw = C.MOTION_COLORS['gt_hist']
    fut_color, fut_lw = C.MOTION_COLORS['gt_hist_missing']
    fut_linestyle = 'dotted'

    gt_hists, gt_futs = agent_sequences[:, :hist_len], agent_sequences[:, hist_len:]
    zipped = zip(gt_hists, gt_futs, agent_types, agent_masks, agent_ids)
    for n, (gt_hist, gt_fut, agent_type, mask, agent_id) in enumerate(zipped):

        gt_hist, gt_fut = gt_hist[mask[:hist_len]], gt_fut[mask[hist_len:]]
        
        # This shouldn't even happen
        if gt_hist.shape[0] == 0 and gt_fut.shape[0] == 0:
            continue

        # Plot History
        if not gt_hist.shape[0] == 0:
            fut_color, fut_lw = C.MOTION_COLORS['gt_future'][0], C.MOTION_COLORS['gt_future'][1]
            fut_linestyle = 'dashed'

            gt_hist_ll = C.reproject_sequences(
                gt_hist[:, 1:], projection) if reproject else gt_hist[:, 1:]
            ax.plot(gt_hist_ll[:, 1], gt_hist_ll[:, 0], color=hist_color, lw=hist_lw) 

            # Plot Agent at Current Timestep (t=H) 
            lon, lat, heading = gt_hist_ll[-1, 1], gt_hist_ll[-1, 0], gt_hist[-1, 0]
            if lon == 0 or lat == 0:
                continue

            # Plot agent icon on last point of sequence. If it's a benchmark agent add halo.
            agent_type = int(agent_type)
            icon = agents[agent_type]
            img = C.plot_agent(icon, heading, zoom=C.ZOOM[agent_type])
            if agent_id in agents_interest:
                color = cm.autumn(halo_values) # '#F29C3A'
                ax.scatter(lon, lat, color=color, alpha=C.MOTION_COLORS['ego_agent'][1], s = 160)        
            ab = AnnotationBbox(img, (lon, lat), frameon = False) 
            ax.add_artist(ab)
        
        # Plot the future
        if not gt_fut.shape[0] == 0:
            gt_fut_ll = C.reproject_sequences(gt_fut[:, 1:], projection) if reproject else gt_fut[:, 1:]
            ax.plot(gt_fut_ll[:, 1], gt_fut_ll[:, 0], color=fut_color, lw=fut_lw, linestyle=fut_linestyle) 

        fut_color, fut_lw = C.MOTION_COLORS['gt_hist_missing']
        fut_linestyle = 'dotted'
    
    # Plot the predictions 
    # NOTE: such a mess... needs to be cleaned up.
    mm = C.MOTION_COLORS['multi_modal']
    pred_scores, mus, sigmas, pred_agents = predictions
    B, N, T, H, D = mus.shape
    pred_color, pred_lw = C.MOTION_COLORS['pred']

    N = len(ego_agent_ids)
    preds = torch.zeros(size=(N, T, 2))
    for i, ego_id in enumerate(ego_agent_ids):
        scores, mu, sigma = pred_scores[i], mus[i, ..., :2], sigmas[i, ..., :2]
        
        start_abs = pred_agents[i, ego_id, hist_len-1, G.XY].detach().cpu().numpy()
        start_heading = pred_agents[i, ego_id, hist_len-1, G.HD].detach().cpu().numpy()
        pred[ego_id, :hist_len] = pred_agents[i, ego_id, :hist_len]

        # TODO: fix reprojection error
        for h in range(H):
            # Transform predicted trajectories 
            pred_ll, pred_xy = xy_to_ll(
                mu[:, :, h], start_abs, start_heading, ref_data, geodesic, return_xyabs=True)
            pred = pred_ll[ego_id]
            preds[ego_id, hist_len:] = pred_xy[ego_id]
            mu_p = mu[:, :, h] + torch.sqrt(sigma[:, :, h])
            mu_n = mu[:, :, h] - torch.sqrt(sigma[:, :, h])
            sigma_p = xy_to_ll(mu_p, start_abs, start_heading, ref_data, geodesic)[ego_id]
            sigma_n = xy_to_ll(mu_n, start_abs, start_heading, ref_data, geodesic)[ego_id]

            # Plot
            score = scores[ego_id, h]
            if score > 0.02:
                pred, sigma_p, sigma_n = pred[hist_len:], sigma_p[hist_len:], sigma_n[hist_len:]
                ax.plot(pred[:, 1], pred[:, 0], color=pred_color, lw=pred_lw, alpha=score.item())
                ax.fill_between(
                    pred[:, 1], sigma_n[:, 0], sigma_p[:, 0], lw=mm[1], alpha=score.item() * 0.65,
                    color=mm[0], interpolate=True)

    # TODO: resolve 3D collisions
    # Compute collision (Collision with full traj)
    agent_combinations = list(itertools.combinations(range(N), 2))
    for i, j in agent_combinations:
        agent_i = (preds[i], None, None, None, None, None)
        agent_j = (preds[j], None, None, None, None, None)
        coll = compute_collisions(agent_i, agent_j, collision_threshold=0.5)
        if np.any(coll == 1.0):
            min_t = np.where(coll == 1.0)[0].min()
            plt.suptitle(f"Predicted Trajectories Collide from t={min_t}s", color="red")
    
    # Plot movement
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axis([west - C.OFFSET, east + C.OFFSET, south - C.OFFSET, north + C.OFFSET])

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()