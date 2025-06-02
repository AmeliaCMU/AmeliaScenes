import os
import json
import math
import pickle
import random
import numpy as np
import pandas as pd

import amelia_scenes.utils.common as C
import amelia_scenes.utils.dataset as D
import amelia_scenes.utils.global_masks as G

from tqdm import tqdm
from easydict import EasyDict
from typing import Tuple, List
from joblib import Parallel, delayed

from amelia_scenes.scoring.kinematic import compute_kinematic_scores
from amelia_scenes.scoring.critical import compute_simple_scene_critical
from amelia_scenes.scoring.interactive import compute_interactive_scores
from amelia_scenes.scoring.crowdedness import compute_simple_scene_crowdedness


class SceneProcessor:
    """ Dataset class for pre-processing airport surface movement data into scenes. """

    def __init__(self, config: EasyDict) -> None:
        """
        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the
            airport trajectory data.
        """
        super(SceneProcessor, self).__init__()

        # Trajectory configuration
        self.airport = config.airport
        self.in_data_dir = os.path.join(config.in_data_dir, self.airport)
        self.out_data_dir = os.path.join(config.out_data_dir, self.airport)
        os.makedirs(self.out_data_dir, exist_ok=True)
        self.blacklist_dir = os.path.join(config.out_data_dir, 'blacklist')
        os.makedirs(self.blacklist_dir, exist_ok=True)
        self.out_data_summary_dir = os.path.join(config.out_summary_dir, self.airport)
        os.makedirs(self.out_data_summary_dir, exist_ok=True)

        self.parallel = config.parallel
        self.overwrite = config.overwrite
        self.add_scores_meta = config.add_scores_meta

        self.seed = config.seed
        self.extend = True
        self.seq_extent = 1

        self.pred_lens = config.pred_lens
        self.pred_len = max(self.pred_lens)
        self.hist_len = config.hist_len
        self.seq_len = self.hist_len + self.pred_len
        self.skip = config.skip
        self.min_agents = config.min_agents
        self.max_agents = config.max_agents
        self.min_valid_points = config.min_valid_points

        self.n_jobs = config.jobs

        # self.data_summary = EasyDict({
        #     'scores': [],
        #     'num_scenes': 0,
        #     'files_scores': []})

        limits_file = os.path.join(config.assets_dir, self.airport, 'limits.json')
        with open(limits_file, 'r') as fp:
            self.ref_data = EasyDict(json.load(fp))

        graph_data_dir = os.path.join(config.graph_data_dir, self.airport)
        print(f"Loading graph data from: {graph_data_dir}")
        pickle_map_filepath = os.path.join(graph_data_dir, "semantic_graph.pkl")
        with open(pickle_map_filepath, 'rb') as f:
            graph_pickle = pickle.load(f)
            self.hold_lines = graph_pickle['hold_lines'][:, 2:4]

        self.blacklist = []
        blacklist_file = os.path.join(self.blacklist_dir, f"{self.airport}.txt")
        if os.path.exists(blacklist_file) and not self.overwrite:
            with open(blacklist_file, 'r') as f:
                self.blacklist = f.read().splitlines()

        file_list = os.listdir(self.in_data_dir)
        for duplicate in list(set(self.blacklist) & set(file_list)):
            file_list.remove(duplicate)
        self.data_files = [os.path.join(self.in_data_dir, f) for f in file_list if f.endswith('.csv')]
        random.seed(self.seed)
        random.shuffle(self.data_files)
        self.data_files = self.data_files[:int(len(self.data_files) * config.perc_process)]

    def process_data(self) -> None:
        """ Processes the CSV data files containing airport trajectory information, creating shards
        containing scenario-level pickle data. If self.parallel is True, it  will process the data
        in parallel, otherwise it will do it sequentially. Once the sharding is done, it will return
        a list containing all generated scenarios and will blacklist any CSV files that failed to
        generate scenarios for the specified conditions.
        """
        print(f"Processing data for airport {self.airport.upper()}.")
        if self.parallel:
            scenes = Parallel(n_jobs=self.n_jobs, require="sharedmem")(
                delayed(self.process_file)(f) for f in tqdm(self.data_files))
            # Unpacking results
            for i in range(len(scenes)):
                res = scenes.pop()
                if res is None:
                    continue
                self.blacklist += res
            del scenes
        else:
            for f in tqdm(self.data_files):
                res = self.process_file(f)
                if res is None:
                    continue
                self.blacklist += res

        # Once all of the data has been processed and the blacklists collected, save them.
        blacklist_file = os.path.join(self.blacklist_dir, f'{self.airport}.txt')
        with open(blacklist_file, 'w') as fp:
            fp.write('\n'.join(self.blacklist))

        # get data percentiles
        # self.data_percentiles()

    def process_file(self, f: str) -> Tuple[List, List, List, List, List, List]:
        """ Processes a single data file. It first obtains the number of possible sequences (given
        the parameters in the configuration file) and then generates scene-level pickle files with
        the corresponding scene's information.

        Inputs
        ------
            f[str]: name of the file to shard.
        """
        print(f"Processing file: {f}")
        base_name = f.split('/')[-1]
        shard_name = base_name.split('.')[0]
        airport_id = base_name.split('_')[0].lower()
        file_time = D._get_file_timestamp(base_name)
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

        blacklist = []
        num_sequences = int(math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))
        if num_sequences < 1:
            blacklist.append(f.removeprefix(self.in_data_dir+'/'))
            return blacklist

        sharded_files = []
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
            time_meta = D._process_timestamp(
                scene_ts=file_time,
                frame_idx=i,
                airport_code=airport_id
            )
            scene = {
                'scenario_id': scenario_id,
                'num_agents': num_agents,
                'airport_id': airport_id,
                'agent_sequences': seq,
                'agent_ids': agent_id,
                'agent_types': agent_type,
                'agent_masks': agent_mask,
                'agent_valid': agent_valid,
                'time_meta': time_meta,
            }
            scene['meta'] = None
            if self.add_scores_meta:
                scene['meta'] = self.process_scores(scene)
                score = scene['meta']['scene_scores']['critical']
                self.data_summary.scores += [score]
                self.data_summary.num_scenes += 1
                self.data_summary.files_scores += [(score, os.path.join(
                    shard_name, f"{scenario_id}_n-{num_agents}.pkl"))]

            scene_filepath = os.path.join(data_dir, f"{scenario_id}_n-{num_agents}.pkl")
            with open(scene_filepath, 'wb') as f:
                pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)

            valid_seq += 1
            sharded_files.append(scene_filepath)

        # If directory is empty, remove it.
        if len(os.listdir(data_dir)) == 0:
            blacklist.append(f.removeprefix(self.in_data_dir+'/'))
            os.rmdir(data_dir)
        return blacklist

    def process_seq(
        self, frame_data: pd.DataFrame, frames: list, seq_idx: int, airport_id: str
    ) -> np.array:
        """ Processes all valid agent sequences.

        Inputs:
        -------
            frame_data[pd.DataFrame]: dataframe containing the scene's trajectory information in the
            following format:
                <FrameID, AgentID, Altitude, Speed, Heading, Lat, Lon, Range, Bearing, AgentType,
                 Interp, x, y>
            frames[list]: list of frames to process.
            seq_idx[int]: current sequence index to process.

        Outputs:
        --------
            seq[np.array]: numpy array containing all processed scene's sequences
            agent_id_list[list]: list with the agent IDs that were processed.
            agent_type_list[list]: list containing the type of agent (Aircraft = 0, Vehicle = 1,
            Unknown=2)
        """
        none_outs = (None, None, None, None, None)
        # All data for the current sequence: from the curr index i to i + sequence length
        seq_data = np.concatenate(frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        # If the speed of all agents is zero or close, return None
        if math.isclose(seq_data[:, G.RAW_IDX.Speed].sum(), 0):
            return none_outs

        # If there are no aircraft in the sequence, return None
        if not np.isin(seq_data[:, G.RAW_IDX.Type].astype(int), C.AIRCRAFT).sum():
            return none_outs

        # IDs of agents in the current sequence
        unique_agents = np.unique(seq_data[:, G.RAW_IDX.ID])
        num_agents = len(unique_agents)
        if num_agents < self.min_agents or num_agents > self.max_agents:
            return none_outs

        num_agents_considered = 0
        seq = np.zeros((num_agents, self.seq_len, G.DIM))
        agent_masks = np.zeros((num_agents, self.seq_len)).astype(bool)
        agent_id_list, agent_type_list, valid_agent_list = [], [], []

        alt_idx = G.RAW_IDX.Altitude

        for _, agent_id in enumerate(unique_agents):
            # Current sequence of agent with agent_id
            agent_seq = seq_data[seq_data[:, 1] == agent_id]

            # Start frame for the current sequence of the current agent reported to 0
            pad_front = frames.index(agent_seq[0, 0]) - seq_idx

            # End frame for the current sequence of the current agent: end of current agent path in
            # the current sequence. It can be sequence length if the aircraft appears in all frames
            # of the sequence or less if it disappears earlier.
            pad_end = frames.index(agent_seq[-1, 0]) - seq_idx + 1

            # Scale altitude
            mx = self.ref_data.limits.Altitude.max
            mn = self.ref_data.limits.Altitude.min
            agent_seq[:, alt_idx] = (agent_seq[:, alt_idx] - mn) / (mx - mn)

            agent_id_list.append(int(agent_id))
            # TODO: fix this. Agent type is not necessarily fixed for the entire trajectory.
            agent_type_list.append(int(agent_seq[0, G.RAW_IDX.Type]))

            # Interpolated mask
            mask = agent_seq[:, G.RAW_IDX.Interp] == '[ORG]'
            # Not interpolated --> Valid
            agent_seq[mask, G.RAW_IDX.Interp] = 1.0
            # Interpolated --> Not valid
            agent_seq[~mask, G.RAW_IDX.Interp] = 0.0

            # Check if there's at least two valid points in the history segment, two valid points in
            # partial segment and two valid points in the future segment
            valid = mask[:self.hist_len].sum() >= self.min_valid_points
            if valid:
                for t in self.pred_lens:
                    if mask[self.hist_len:self.hist_len+t].sum() < self.min_valid_points:
                        valid = False
                        break
            valid_agent_list.append(valid)

            # TODO: Impute needs to be debugged. Imputing should not happen since it was done in SWIM.
            agent_seq = C.impute(agent_seq, pad_end - pad_front)  # self.seq_len)
            valid_mask = agent_seq[:, G.RAW_IDX.Interp].astype(bool)
            agent_masks[num_agents_considered, pad_front:pad_end] = valid_mask

            agent_seq = agent_seq[:, G.RAW_SEQ_MASK]
            seq[num_agents_considered, pad_front:pad_end] = agent_seq[:, G.SEQ_ORDER]
            num_agents_considered += 1

        # Return Nones if there aren't any valid agents
        valid_agent_list = np.asarray(valid_agent_list)
        if valid_agent_list.sum() == 0:
            return none_outs

        # Return Nones if the number of considered agents is less than the required
        if num_agents_considered < self.min_agents:
            return none_outs

        return seq[:num_agents_considered], agent_id_list, agent_type_list, valid_agent_list, agent_masks[:num_agents_considered]

    def process_scores(self, scene):
        """ Computes kinematic and interactive scores for all valid agent sequences.

        Inputs:
        -------
            scene[dict]: dictionary containing agent sequences and meta information.

        Outputs:
        --------
            scores[dict]: dictionary containing individual and interactive scores for each agent,
            and scene scores.
        """
        crowd_scene_score = compute_simple_scene_crowdedness(scene, self.max_agents)
        kin_agents_scores, kin_scene_score = compute_kinematic_scores(scene, self.hold_lines)
        int_agents_scores, int_scene_score = compute_interactive_scores(scene, self.hold_lines)
        crit_agent_scores, crit_scene_score = compute_simple_scene_critical(
            agent_scores_list=[kin_agents_scores.copy(), int_agents_scores].copy(),
            scene_score_list=[crowd_scene_score.copy(), kin_scene_score.copy(), int_scene_score.copy()]
        )
        return {
            'agent_scores': {
                'kinematic': kin_agents_scores,
                'interactive': int_agents_scores,
                'critical': crit_agent_scores
            },
            'agent_order': {
                'random': C.get_random_order(scene['num_agents'], scene['agent_valid'], self.seed),
                # 'kinematic': C.get_sorted_order(kin_agents_scores),
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

    # def data_percentiles(self):

    #     scores = np.array(self.data_summary.scores)
    #     percentiles = [50, 60, 70, 80, 90, 95, 99, 99.5]
    #     percentile_values = np.percentile(scores, percentiles)

    #     scenes_data = self.data_summary.files_scores
    #     data_summary = {"num_scenes": self.data_summary.num_scenes,
    #                     "percentile_scores": {},
    #                     }
    #     for i, p in enumerate(percentiles):
    #         threshold = percentile_values[i]
    #         filtered = [{scene_file: score} for (score, scene_file) in scenes_data if score >= threshold]
    #         data_summary["percentile_scores"][str(p)] = {
    #             "threshold": float(threshold),
    #             "num_scenes": len(filtered),
    #             "scenes": filtered
    #         }

    #     summary_file = os.path.join(self.out_data_summary_dir, f"{self.airport}_summary.json")
    #     with open(summary_file, 'w') as f:
    #         json.dump(data_summary, f, indent=4)
    #     print(f"Dataset summary saved to {summary_file}")
