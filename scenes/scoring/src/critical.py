import numpy as np

def compute_simple_scene_critical(agent_scores_list: list, scene_score_list: list):
    """ Computes a simple score for the scene crowdedness consisting of the average weight value 
    for the agents in the scene:
                        W = [w_i forall i in agent_types]
                        C = mean(W) * sum(W)
    """
    for i in range(1, len(agent_scores_list)):
        agent_scores_list[0] += agent_scores_list[i] 
    scene_score = np.asarray(scene_score_list).sum()
    return agent_scores_list[0], scene_score