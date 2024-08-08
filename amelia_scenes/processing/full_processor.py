import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import random

import amelia_scenes.utils.global_masks as G
import amelia_scenes.utils.common as C

from easydict import EasyDict
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Tuple, List

from amelia_scenes.scoring.crowdedness import compute_simple_scene_crowdedness
from amelia_scenes.scoring.kinematic import compute_kinematic_scores
from amelia_scenes.scoring.interactive import compute_interactive_scores
from amelia_scenes.scoring.critical import compute_simple_scene_critical

from amelia_scenes.processing.scene_processor import SceneProcessor

class FullProcessor(SceneProcessor):
    """ Dataset class for pre-processing airport surface movement data into scenes. """

    def __init__(self, config: EasyDict) -> None:
        """
        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the
            airport trajectory data.
        """
        super(FullProcessor, self).__init__(config=config)

    def process_file(self, f: str) -> Tuple[List, List, List, List, List, List]:
        """ Processes a single data file. It first obtains the number of possible sequences (given
        the parameters in the configuration file) and then generates scene-level pickle files with
        the corresponding scene's information.

        Inputs
        ------
            f[str]: name of the file to shard.
        """
        shard_name = f.split('/')[-1].split('.')[0]
        airport_id = f.split('/')[-1].split('_')[0].lower()
        data_dir = os.path.join(self.out_data_dir, shard_name)

        # Check if the file has been sharded already. If so, add sharded files to the scenario list.
        if not self.overwrite and (os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0):
            return None

        # Otherwise, shard the file and add it to the scenario list.
        data = pd.read_csv(f)

        # Get the number of unique frames
        frames = data.Frame.unique().tolist()
        frame_data = []
        for frame_num in frames:
            frame = data[:][data.Frame == frame_num]
            frame_data.append(frame)

        num_sequences = int(
            math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))
        if num_sequences < 1:
            return None

        sharded_files = []
        blacklist = []
        os.makedirs(data_dir, exist_ok=True)

        valid_seq = 0
        for i in range(0, num_sequences * self.skip + 1, self.skip):
            scenario_id = str(valid_seq).zfill(6)
            seq, agent_id, agent_type, agent_valid, agent_mask = self.process_seq(
                frame_data=frame_data, frames=frames, seq_idx=i, airport_id=airport_id)
            if seq is None:
                continue

            # Get agent array based on random and safety criteria
            num_agents, _, _ = seq.shape

            scene = EasyDict({
                'scenario_id': scenario_id,
                'num_agents': num_agents,
                'airport_id': airport_id,
                'agent_sequences': seq,
                'agent_ids': agent_id,
                'agent_types': agent_type,
                'agent_masks': agent_mask,
                'agent_valid': agent_valid,
            })

            crowd_scene_score = compute_simple_scene_crowdedness(
                scene, self.max_agents)
            kin_agents_scores, kin_scene_score = compute_kinematic_scores(
                scene, self.hold_lines)
            int_agents_scores, int_scene_score = compute_interactive_scores(
                scene, self.hold_lines)
            crit_agent_scores, crit_scene_score = compute_simple_scene_critical(
                agent_scores_list=[kin_agents_scores, int_agents_scores],
                scene_score_list=[crowd_scene_score,
                                  kin_scene_score, int_scene_score]
            )

            scene['meta'] = {
                'agent_scores': {
                    'kinematic': kin_agents_scores,
                    'interactive': int_agents_scores,
                    'critical': crit_agent_scores
                },
                'agent_order': {
                    'random': C.get_random_order(scene.num_agents, scene.agent_valid, self.seed),
                    'kinematic': C.get_sorted_order(kin_agents_scores),
                    'interactive': C.get_sorted_order(int_agents_scores),
                    'critical': C.get_sorted_order(crit_agent_scores)
                },
                'scene_scores': {
                    'crowdedness': crowd_scene_score,
                    'kinematic': kin_scene_score,
                    'interactive': int_scene_score,
                    'critical': crit_scene_score
                },
            }

            scenario_filepath = os.path.join(
                data_dir, f"{scenario_id}_n-{num_agents}.pkl")
            with open(scenario_filepath, 'wb') as f:
                pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)

            valid_seq += 1
            sharded_files.append(scenario_filepath)

        # If directory is empty, remove it.
        if len(os.listdir(data_dir)) == 0:
            blacklist.append(f.removeprefix(self.in_data_dir+'/'))
            os.rmdir(data_dir)
        return blacklist
