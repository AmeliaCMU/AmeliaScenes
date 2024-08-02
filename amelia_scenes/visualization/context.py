import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, List

from amelia_scenes.visualization.common import COLOR_CODES


def plot_context(context, ax, xyminmax=None) -> None:
    """ Visualization tool to plot the map context.

    Inputs
    ------
        context[np.array]: vectors to plot
        ax[plt.axis]: matplotlib axis to plot on.
    """
    for pl in context:
        if len(pl) == 5:
            y_start_rl, x_start_rl, y_end_rl, x_end_rl, semantic_id = pl  # Pre-onehot encoding
        # Post-onehot encoding
        else:
            x_start_rl = pl[1]
            y_start_rl = pl[0]
            x_end_rl = pl[3]
            y_end_rl = pl[2]

        if x_start_rl == 0 and x_end_rl == 0 and y_start_rl == 0 and y_end_rl == 0:
            continue

        if len(pl) == 5:
            color = COLOR_CODES[semantic_id]
        else:
            color_id = (np.where(pl == 1)[0][0] - 4) + 1
            color = COLOR_CODES[color_id]

        if xyminmax is not None:
            xmin, ymin, xmax, ymax = xyminmax
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.set_aspect('equal')
        ax.plot([x_start_rl, x_end_rl], [y_start_rl, y_end_rl],
                color=color, linewidth=1, alpha=1)


def debug_plot(
    ego_id: int, rel_seq: np.array, full_context: np.array, local_context_list: List[np.array],
    llimits: Tuple
) -> None:
    """ Simple visualization tool to debug global vs extracted local contexts with their
    corresponding trajectories.

    Inputs
    ------
        ego_id[int]: id of the ego agent for the current scene.
        rel_seq[np.array(N, T, D)]: scene containing the transformed trajectories w.r.t. ego agent.
        full_context[np.array]: global context (map) transformed w.r.t. ego agent.
        local_context_list[List[np.array]]: list of local context for each of the agents.
    """
    # Plot Rotated Map
    fig, ax = plt.subplots()
    x_min, y_min, x_max, y_max = llimits
    fig.set_size_inches(8, 8)

    ax.set_title('Global Scene')
    plot_context(full_context, ax)

    # Plot all relative rel_seq
    num_agents, _, _ = rel_seq.shape
    for n in range(num_agents):
        color = 'r' if n == ego_id else 'b'
        X, Y = rel_seq[n, :, 1], rel_seq[n, :, 0]
        ax.scatter(X, Y, marker='H', color=color)

    # Plot patches in separate figures
    for i, patch in enumerate(local_context_list):
        fig, ax_patch = plt.subplots()
        color = 'r' if i == ego_id else 'b'
        X, Y = rel_seq[i, :, 1], rel_seq[i, :, 0]
        ax_patch.scatter(X, Y, marker='H', color=color)
        plot_context(patch, ax_patch)

    plt.show()
    # plt.close()
