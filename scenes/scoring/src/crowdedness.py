import numpy as np

from easydict import EasyDict

from scenes.utils.common import WEIGHTS, AIRCRAFT


def compute_simple_scene_crowdedness(scene: EasyDict, max_agents: int):
    """ Computes a simple score for the scene crowdedness consisting of the average weight value
    for the agents in the scene:
                        W = [w_i forall i in agent_types]
                        C = mean(W) * sum(W)
    """
    crowdedness_score = np.asarray(
        [WEIGHTS[agent_type] for agent_type in scene.agent_types])
    norm_constant = max_agents * WEIGHTS[AIRCRAFT]
    return crowdedness_score.sum() / norm_constant
