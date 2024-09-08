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
    scene: dict, assets: Tuple, benchmark: dict, filename: str = 'temp.png', dpi=600, 
    reproject: bool = False, projection: str = 'EPSG:3857'
) -> None:
    """ Visualizing benchmark scenes. Like 'plot_scene_simple' but adds benchmark metadata """
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.5) 
    
    agents_interest = benchmark['bench_agents']
    halo_value = benchmark['timestep']
    airport_name = benchmark['airport'].values[0]
    date = benchmark['date'].values[0]
    ax.set_title(f"{airport_name.upper()} ({date})")

    C.plot_sequences(ax, scene, agents, agents_interest, halo_value=halo_value, reproject=reproject)
    C.save(ax, filename, dpi, limits=[west, east, south, north], force_extent=True)

def plot_scene_benchmark_predictions(
    scene: Tuple, assets: Tuple, benchmark: dict, predictions: Tuple, filename: str = 'temp', 
    dpi=600, reproject: bool = False, projection: str = 'EPSG:3857', score_thresh: float = 0.1
) -> None:
    """ Visualizing benchmark scenes. Like 'plot_scene_simple' but adds benchmark metadata """
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    geodesic = Geodesic.WGS84
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    fig, ax = plt.subplots()

    # Display global map
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.5) 
    
    hist_len, ego_agent_ids = scene['hist_len'], scene['ego_agent_ids']
    agents_interest = benchmark['bench_agents']
    halo_value = benchmark['timestep']
    airport_name = benchmark['airport'].values[0]
    date = benchmark['date'].values[0]
    plt.title(f"{airport_name.upper()} ({date})")

    # Plots sequences segmented as 'history', 'future' and 'future entering'
    C.plot_sequences_segmented(
        ax, scene, agents, agents_interest, halo_value, reproject=reproject, projection=projection)
    
    # NOTE: Plot the predictions. For now, this is too specific to benchmark requirements but should
    # modularize later
    pred_scores, mus, sigmas, pred_agents = predictions
    mm = C.MOTION_COLORS['multi_modal']
    B, N, T, H, D = mus.shape
    pred_color, pred_lw = C.MOTION_COLORS['pred']

    N = len(ego_agent_ids)
    preds_xy, preds_ll = torch.zeros(size=(N, T, 2)), torch.zeros(size=(N, T, 2))
    for i, ego_id in enumerate(ego_agent_ids):
        scores, mu, sigma = pred_scores[i], mus[i, ..., :2], sigmas[i, ..., :2]
        
        start_abs = pred_agents[i, ego_id, hist_len-1, G.XY].detach().cpu().numpy()
        start_heading = pred_agents[i, ego_id, hist_len-1, G.HD].detach().cpu().numpy()
        preds_xy[ego_id, :hist_len] = pred_agents[i, ego_id, :hist_len, G.XY]
        preds_ll[ego_id, :hist_len] = pred_agents[i, ego_id, :hist_len, G.LL]

        # TODO: fix reprojection error
        best_h = scores[ego_id].argmax().item()
        for h in range(H):
            # Transform predicted trajectories 
            pred_ll, pred_xy = xy_to_ll(
                mu[:, :, h], start_abs, start_heading, ref_data, geodesic, return_xyabs=True)
            if h == best_h:
                preds_ll[ego_id, hist_len:] = pred_ll[ego_id, hist_len:]
                preds_xy[ego_id, hist_len:] = pred_xy[ego_id, hist_len:]
            
            mu_p = mu[:, :, h] + torch.sqrt(sigma[:, :, h])
            mu_n = mu[:, :, h] - torch.sqrt(sigma[:, :, h])
            sigma_p = xy_to_ll(mu_p, start_abs, start_heading, ref_data, geodesic)[ego_id]
            sigma_n = xy_to_ll(mu_n, start_abs, start_heading, ref_data, geodesic)[ego_id]

            # Plot
            score = scores[ego_id, h].item()
            if score > score_thresh:
                pred = pred_ll[ego_id, hist_len:]
                pred = C.reproject_sequences(pred, projection) if reproject else pred
                sigma_p = C.reproject_sequences(sigma_p, projection) if reproject else sigma_p
                sigma_n = C.reproject_sequences(sigma_n, projection) if reproject else sigma_n
                sigma_p, sigma_n = sigma_p[hist_len:], sigma_n[hist_len:]
                ax.plot(pred[:, 1], pred[:, 0], color=pred_color, lw=pred_lw, alpha=score)
                ax.fill_between(
                    pred[:, 1], sigma_n[:, 0], sigma_p[:, 0], lw=mm[1], alpha=score * 0.50, 
                    color=mm[0], interpolate=True)

    # TODO: resolve 3D collisions
    # Compute collision (Collision with full traj)
    agent_combinations = list(itertools.combinations(range(N), 2))
    for i, j in agent_combinations:
        agent_i = (preds_xy[i], None, None, None, None, None)
        agent_j = (preds_xy[j], None, None, None, None, None)
        coll = compute_collisions(agent_i, agent_j, collision_threshold=0.2)#[hist_len:]
        if np.any(coll == 1.0):
            min_t = np.where(coll == 1.0)[0].min()# + hist_len
            plt.suptitle(f"Predicted Trajectories Collide from t={min_t}s", color="red")
            filename += "_coll"
            pred_i, pred_j = preds_ll[i, min_t].unsqueeze(0), preds_ll[j, min_t].unsqueeze(0)
            pred_i = C.reproject_sequences(pred_i, projection) if reproject else pred_i
            ax.scatter(pred_i[:, 1], pred_i[:, 0], marker=(10, 1, 0), color='red', s = 70, zorder=10)
            ax.scatter(pred_i[:, 1], pred_i[:, 0], marker=(10, 1, 2), color='orange', s = 20, zorder=10)
            
            pred_j = C.reproject_sequences(pred_j, projection) if reproject else pred_j
            ax.scatter(pred_j[:, 1], pred_j[:, 0], marker=(10, 1, 0), color='red', s = 70, zorder=10)
            ax.scatter(pred_j[:, 1], pred_j[:, 0], marker=(10, 1, 2), color='orange', s = 20, zorder=10)
            
    # Plot movement
    C.save(ax, filename, dpi, limits=[west, east, south, north], force_extent=True)