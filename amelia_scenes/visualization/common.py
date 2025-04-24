import glob
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from natsort import natsorted
from PIL import Image
from pyproj import Transformer
from scipy import ndimage
from typing import Tuple

from amelia_scenes.utils import global_masks as G


# -------------------------------------------------------------------------------------------------#
#                                   COMMON GLOBAL VARIABLES                                        #
# -------------------------------------------------------------------------------------------------#
MAP_CRS = 'EPSG:4326'

STYLES = {
    'family': 'monospace',
    'color':  'black',
    'weight': 'bold',
}

OFFSET = 0.008
EPS = 1e-5

COLOR_CODES = {
    1: '#ff9b21',
    2: '#fc0349',
    3: '#1fe984',
    4: '#6285cf',
    5: '#f05eee',
}


MOTION_COLORS = {
    'gt_hist': ('#FF5A4C', 0.9),
    'gt_hist_missing': ('#1289F3', 0.9),
    'gt_future': ('#8B5FBF', 0.9),
    'multimodal': ('#14F5B2', 1.0),
    'multi_modal': ((20/255, 245/255, 178/255), 1.0),
    'pred': ('#07563F', 0.9),
    'ego_agent': ('#0F94D2', 0.5),
    'interest_agent': ('#F29C3A', 0.5),
    'other_agent': ('#F29C3A', 0.0),
}

OTHER = -1
AIRCRAFT = 0
VEHICLE = 1
UNKNOWN = 2
AIRCRAFT_PADDED = 3
AIRCRAFT_INVALID = 4


KNOTS_TO_MPS = 0.51444445
KNOTS_TO_KPH = 1.852
HOUR_TO_SECOND = 3600
KMH_TO_MS = 1/3.6

ZOOM = {
    OTHER: 0.025,
    AIRCRAFT: 0.015,
    VEHICLE: 0.2,
    UNKNOWN: 0.015,
    AIRCRAFT_PADDED: 0.015,
    AIRCRAFT_INVALID: 0.015,
}

AGENT_COLORS = {
    OTHER: '#F29C3A',
    AIRCRAFT: '#FF5A4C',
    VEHICLE: '#5ea2cf',
    UNKNOWN: '#8a8585',
    AIRCRAFT_PADDED: '#2725e1',
    AIRCRAFT_INVALID: '#7525e1',
}

LL_TO_KNOTS = 1000 * 111 * 1.94384

# -------------------------------------------------------------------------------------------------#
#                                      COMMON PLOT UTILS                                           #
# -------------------------------------------------------------------------------------------------#


def norm(arr, method: str = 'minmax'):
    assert method in ['minmax', 'meanstd']
    if np.all(arr == 0.0):
        return arr

    if method == 'minmax':
        if (arr.max() - arr.min()) != 0.0:
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            arr /= arr.max()
    else:
        if arr.std() != 0.0:
            arr = (arr - arr.min()) / arr.std()
    return arr


def plot_agent(asset, heading, zoom=0.015, alpha=1.0):
    img = ndimage.rotate(asset, heading)
    img = np.fliplr(img)
    img = OffsetImage(img, zoom=zoom, alpha=alpha)
    return img


def plot_faded_prediction(lon: np.array, lat: np.array, axis=None, gradient=None, color=None):
    points = np.vstack((lon, lat)).T.reshape(-1, 1, 2)
    segments = np.hstack((points[:-1], points[1:]))
    lc = LineCollection(segments, lw=0.35, array=gradient, cmap=color)
    axis.add_collection(lc)
    del lc


def get_velocity(trajectory: np.array):
    # Calculate displacement vector
    displacement = trajectory[1:] - trajectory[0:-1]
    # Use norm to get magnitude of displacement and take mean to get speed
    speed = np.linalg.norm(displacement, axis=1).mean() * LL_TO_KNOTS
    return speed


def plot_speed_histogram(ax, vel_mu, vel_sigma, vel_hist, vel_fut):
    ax.set_xlim([0, 140])

    # Plot velocity distribution
    freq, _, patches = ax.hist(
        vel_sigma, bins=10, color=MOTION_COLORS['multi_modal'][0], edgecolor="k",
        linewidth=0.5, alpha=1, density=True)
    ax.axvline(vel_hist, 0, 1000, color=MOTION_COLORS['gt_hist'][0], linewidth=1.9)
    ax.axvline(vel_fut, 0, 1000, color=MOTION_COLORS['gt_future'][0], linewidth=1.9)
    ax.text(
        vel_hist, max(freq), "H", rotation=90, verticalalignment='center',
        color=MOTION_COLORS['gt_hist'][0])
    ax.text(
        vel_fut, max(freq), "G", rotation=90, verticalalignment='center',
        color=MOTION_COLORS['gt_future'][0])

    # Iterate through predicted speed
    for i, v in enumerate(vel_mu):
        ax.axvline(v, 0, 1000, color=MOTION_COLORS['pred'][0], linewidth=1.9)

    ax.set_yticks([])
    ax.set_xticks([0, 35, 70, 105, 140])
    ax.set_xlabel("Knots")
    ax.set_title("Speed Histogram", fontdict=STYLES)
    ax.grid(True)


def transform_extent(extent, original_crs: str, target_crs: str):
    transformer = Transformer.from_crs(original_crs, target_crs)
    north, east, south, west, _, _ = extent
    xmin_trans, ymin_trans = transformer.transform(south, west)
    xmax_trans, ymax_trans = transformer.transform(north, east)
    return (ymax_trans, xmax_trans, ymin_trans, xmin_trans)


def reproject_sequences(sequence, target_projection):
    if sequence.shape[0] > 1:
        lat = np.array(sequence[:, 0::2])
        lon = np.array(sequence[:, 1::2])
    else:
        lat = np.array(sequence[:, 0])
        lon = np.array(sequence[:, 1])

    projector = Transformer.from_crs(MAP_CRS, target_projection)
    x, y = projector.transform(lat, lon)
    transformed_sequence = []
    for x, y in zip(x, y):
        transformed_sequence.append(y)
        transformed_sequence.append(x)

    transformed_sequence = np.array(transformed_sequence)
    T, D = sequence.shape

    transformed_sequence = transformed_sequence.reshape((T, D))
    return transformed_sequence


def save(
    ax, filename: str = "temp.png", dpi: int = 200, clear_ticks: bool = True,
    force_extent: bool = False, limits: Tuple = None,
) -> None:
    if clear_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if force_extent:
        xmin, xmax, ymin, ymax = limits
        ax.axis([xmin - OFFSET, xmax + OFFSET, ymin - OFFSET, ymax + OFFSET])

    # Set figure bbox around the predicted trajectory
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_sequences(
    ax, scene: dict, agents: dict, agents_interest: list = [], halo_values: list = [],
    reproject: bool = False, projection: str = 'EPSG:3857'
) -> None:
    agent_sequences, agent_masks = scene['agent_sequences'][:, :, G.HLL], scene['agent_masks']
    agent_types, agent_ids, agent_valid = scene['agent_types'], scene['agent_ids'], scene['agent_valid']

    num_agents, seq_len, _ = agent_sequences.shape
    # halo values check
    if agents_interest and not halo_values:
        halo_values = [MOTION_COLORS['interest_agent']] * len(agents_interest)
    agents_plot = {agent_id: 1.0-halo_value for agent_id, halo_value in zip(agents_interest, halo_values)}
    
    # Display each trajectory
    traj_color, traj_lw = MOTION_COLORS['gt_hist'][0], MOTION_COLORS['gt_hist'][1]
    zipped = zip(agent_sequences, agent_types, agent_masks, agent_ids, agent_valid)
    for n, (trajectory, agent_type, mask, agent_id, valid) in enumerate(zipped):
        traj = trajectory[mask]
        if traj.shape[0] == 0:
            continue

        # Get heading at last point of trajectory history.
        heading = traj[-1, 0]
        traj_ll = reproject_sequences(traj[:, 1:], projection) if reproject else traj[:, 1:]
        lon, lat = traj_ll[-1, 1], traj_ll[-1, 0]
        if lon == 0 or lat == 0:
            continue
        
        # TODO: Need to check for all agent types **sigh**
        agent_type, alpha = int(agent_type), 1.0
        alpha = 1.0 if valid else 0.3 
        traj_ls = 'solid' if valid and mask.sum() == seq_len else 'dotted'
        
        # Place plane on last point of ground truth sequence
        icon = agents[agent_type]
        img = plot_agent(icon, heading, zoom=ZOOM[agent_type], alpha=alpha)
        if agent_id in agents_interest:
            alpha = agents_plot[agent_id]
            ax.scatter(lon, lat, color='#FF5A4C', alpha=alpha, s=160)
        ab = AnnotationBbox(img, (lon, lat), frameon=False)
        ax.add_artist(ab)

        ax.plot(traj_ll[:, 1], traj_ll[:, 0], color=traj_color, lw=traj_lw, ls=traj_ls, alpha=alpha)
        # ax.text(traj_ll[0, 1], traj_ll[0, 0], f"{agent_id}")

def plot_sequences_segmented(
    ax, scene: dict, agents: dict, agents_interest: list = [], halo_values: list = [],
    reproject: bool = False, projection: str = 'EPSG:3857'
) -> None:
    # breakpoint()
    agent_sequences, agent_masks = scene['agent_sequences'][:, :, G.HLL], scene['agent_masks']
    agent_types, agent_ids, hist_len = scene['agent_types'], scene['agent_ids'], scene['hist_len']

    hist_color, hist_lw = MOTION_COLORS['gt_hist']
    fut_color, fut_lw = MOTION_COLORS['gt_hist_missing']
    fut_linestyle = 'dotted'

    # halo values check
    # if agents_interest.size and not halo_values.size:
        # halo_values = [MOTION_COLORS['interest_agent']] * len(agents_interest)
    # agents_plot = {agent_id: halo_value for agent_id, halo_value in zip(agents_interest, halo_values)}

    gt_hists, gt_futs = agent_sequences[:, :hist_len], agent_sequences[:, hist_len:]
    zipped = zip(gt_hists, gt_futs, agent_types, agent_masks, agent_ids)
    for n, (gt_hist, gt_fut, agent_type, mask, agent_id) in enumerate(zipped):

        gt_hist, gt_fut = gt_hist[mask[:hist_len]], gt_fut[mask[hist_len:]]

        # This shouldn't even happen
        if gt_hist.shape[0] == 0 and gt_fut.shape[0] == 0:
            continue

        # Plot History
        if not gt_hist.shape[0] == 0:
            fut_color, fut_lw = MOTION_COLORS['gt_future'][0], MOTION_COLORS['gt_future'][1]
            fut_linestyle = 'dashed'

            gt_hist_ll = reproject_sequences(
                gt_hist[:, 1:], projection) if reproject else gt_hist[:, 1:]
            ax.plot(gt_hist_ll[:, 1], gt_hist_ll[:, 0], color=hist_color, lw=hist_lw)

            # Plot Agent at Current Timestep (t=H)
            lon, lat, heading = gt_hist_ll[-1, 1], gt_hist_ll[-1, 0], gt_hist[-1, 0]
            if lon == 0 or lat == 0:
                continue

            # Plot agent icon on last point of sequence. If it's a benchmark agent add halo.
            agent_type = int(agent_type)
            icon = agents[agent_type]
            img = plot_agent(icon, heading, zoom=ZOOM[agent_type])
            if agent_id in agents_interest:
                # color = cm.autumn(halo_value) # '#F29C3A'
                color = '#F29C3A'
                ax.scatter(lon, lat, color=color, alpha=MOTION_COLORS['ego_agent'][1], s = 160)        
            ab = AnnotationBbox(img, (lon, lat), frameon = False) 
            # ax.text(lon, lat, scene['agent_ids'][n], fontsize=10)
            ax.add_artist(ab)

        # Plot the future
        if not gt_fut.shape[0] == 0:
            gt_fut_ll = reproject_sequences(gt_fut[:, 1:], projection) if reproject else gt_fut[:, 1:]
            ax.plot(gt_fut_ll[:, 1], gt_fut_ll[:, 0], color=fut_color, lw=fut_lw, linestyle=fut_linestyle)

        fut_color, fut_lw = MOTION_COLORS['gt_hist_missing']
        fut_linestyle = 'dotted'


def to_gif(base_dir, scene_tag):
    files = natsorted(glob.glob(f"{base_dir}/{scene_tag}*.png"))
    imgs = [Image.open(f) for f in natsorted(glob.glob(f"{base_dir}/{scene_tag}*.png"))]
    imgs[0].save(
        f'{base_dir}/{scene_tag}.gif', format='GIF', append_images=imgs[1:], save_all=True,
        duration=200, disposal=2, loop=0)

    for f in files:
        if not "coll" in f:
            os.remove(f)
