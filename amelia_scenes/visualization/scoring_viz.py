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
    mm = C.MOTION_COLORS['multi_modal']

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
            axs[0].scatter(lon, lat, color='#F39C12', alpha=C.MOTION_COLORS['ego_agent'][1], s=160)

        if n in critical_order:
            axs[1].scatter(lon, lat, color='#F39C12', alpha=C.MOTION_COLORS['ego_agent'][1], s=160)

        ab = AnnotationBbox(img, (lon, lat), frameon=False)
        axs[0].add_artist(ab)

        ab = AnnotationBbox(img, (lon, lat), frameon=False)
        axs[1].add_artist(ab)

        axs[0].plot(
            traj_ll[:, 1], traj_ll[:, 0], color=C.MOTION_COLORS['gt_hist'][0],
            lw=C.MOTION_COLORS['gt_hist'][1])
        axs[1].plot(
            traj_ll[:, 1], traj_ll[:, 0], color=C.MOTION_COLORS['gt_hist'][0],
            lw=C.MOTION_COLORS['gt_hist'][1])

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(tag, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_scene_scores(
    scene: dict, assets: Tuple, filename: str, scores: dict = {}, show_scores: bool = False, dpi=600,
    reproject: bool = False, projection: str = 'EPSG:3857', to_scale: bool = False
) -> None:
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)
    if to_scale:
        C.agents_to_scale(agents, limits, bkg)

    # Normalize features
    agent_scores, scene_score = scores['agent_scores'], scores['scene_scores']
    valid_agents = scores['valid_agents']
    agent_scores = C.norm(agent_scores)

    fig, ax = plt.subplots()

    # Plots airport map as background
    ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.3)

    C.plot_sequences_cm(
        ax, scene, agents, reproject=reproject, projection=projection, valid_agents=valid_agents,
        Z=agent_scores, show_scores=show_scores)
    C.save(ax, filename, dpi)


def plot_scene_features(
    scene: dict, assets: Tuple, filename: str, features: dict = {}, dpi=600,
    reproject: bool = False, projection: str = 'EPSG:3857'
) -> None:
    bkg, hold_lines, graph_nx, limits, agents = assets
    limits, ref_data = limits
    north, east, south, west, z_min, z_max = limits
    if reproject:
        north, east, south, west = C.transform_extent(limits, C.MAP_CRS, projection)

    valid_agents = features['agent_idxs']
    features_to_add = [key for key in features.keys() if 'idxs' not in key]
    axs_titles = ['Scene'] + [' '.join(m.split('_')).title() for m in features_to_add]

    nrows, ncols = 2, 1 + len(features_to_add) // 2 + len(features_to_add) % 2
    extra_tiltes = ['Empty' for _ in range(max(0, ncols * nrows - len(axs_titles)))]
    axs_titles += extra_tiltes

    _, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(5 * ncols, 5 * nrows))

    # Normalize features
    Z = {}
    for k in features_to_add:
        v = features[k]
        if 'rate' in k:
            v = [v1 for v1, v2 in v]
        elif 'waiting' in k:
            v = [t for t, d, i in v]
        Z[k] = C.norm(np.asarray(v))

    # sequences = scene['agent_sequences']

    # Plot background assets for all subplots
    for i, ax in enumerate(axs.reshape(-1)):
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(axs_titles[i])
        ax.imshow(bkg, zorder=0, extent=[west, east, south, north], alpha=0.3)

    #     if 'waiting' in axs_titles[i].lower():
    #         ax.scatter(graph_cps[:, 0], graph_cps[:, 1], c='magenta', s=0.1, zorder=1, alpha=0.05)
    #     ax.set_xlim(west-2, east+2)
    #     ax.set_ylim(south-2, north+2)

    # Plots raw scene
    C.plot_sequences(axs[0, 0], scene, agents, reproject=reproject, projection=projection)

    # Plot scenes by feature
    row, col = 0, 1
    for i, k in enumerate(features_to_add):
        if col >= ncols:
            row, col = 1, 0

        C.plot_sequences_cm(
            axs[row, col], scene, agents, reproject=reproject, projection=projection,
            valid_agents=valid_agents, Z=Z[k])
        col += 1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    C.save(ax, filename, dpi)
