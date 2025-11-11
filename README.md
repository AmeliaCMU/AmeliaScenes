# AmeliaScenes

This repository contains the code used in this paper to generate scenes for trajectory forecasting.

***Amelia: A Large Dataset and Model for Airport Surface Movement Forecasting [[paper](https://arxiv.org/pdf/2407.21185)]***

[Ingrid Navarro](https://navars.xyz) *, [Pablo Ortega-Kral](https://paok-2001.github.io) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Haichuan Wang,
Zelin Ye, Jong Hoon Park, [Jean Oh](https://cmubig.github.io/team/jean_oh/) and [Sebastian Scherer](https://theairlab.org/team/sebastian/)

## Overview

**AmeliaScenes**: Tool for generating airport surface movement scenes from raw trajectory data collected with [AmeliaSWIM](https://github.com/AmeliaCMU/AmeliaSWIM). It takes CSV files containing various fields representing agent trajectory data and produces scenes based on certain specifications, including number of desired agents and scene length.

<div align="center">
  <img width="800" src="./assets/kbos.gif" alt="(KBOS)">
  <h5>Scene Example from Boston Logan International Airport (KBOS).</h5>
</div>

**AmeliaScenes** also provides scene and per-agent characterization tools as meta information for each agent's kinematic and interactive profile.

<div align="center">
  <img width="800" src="./assets/scoring.png" alt="Scene Scoring">
  <h5>Scene Scoring Example.</h5>
</div>

Finally, **AmeliaScenes** also provides a dataset splitting script with various `train/val/test` splitting strategies.

## Pre-requisites

### Dataset

To run this repository, you first need to download the amelia dataset. For more details, follow go to the following link [here](https://ameliacmu.github.io/amelia-dataset/). The dataset is hosted in HuggingFace in the in **[AmeliaCMU](https://huggingface.co/AmeliaCMU)** organization. Currenlty three are two versions of the dataset available:

- **(Amelia-10)[https://huggingface.co/datasets/AmeliaCMU/Amelia-10]**: which contains 1 month of data for each of 10 airports.
- **(Amelia42-Mini)[https://huggingface.co/datasets/AmeliaCMU/Amelia42-Mini]**: which contains 15 days of data for each of 42 airports.

In order to download the dataset, git lfs is required. Please follow the instructions [here](https://git-lfs.com) to install git lfs.

To download the Amelia-10 dataset, run the following command:

```bash
git lfs install
git clone https://huggingface.co/datasets/AmeliaCMU/Amelia-10
```

Once downloaded, create a symbolic link into `datasets` from the AmeliaScenes repository:

```bash
cd datasets
ln -s /path/to/Amelia-10/data amelia
```

the resulting structure should look like this:

```bash
|-- AmeliaScenes
    |-- datasets
        |-- amelia
            |-- traj_data_a10v08
                |-- raw_trajectories
                    |-- kbos
                    |-- kdca
                    |-- kewr
                    |-- kjfk
                    |-- klax
                    |-- kmdw
                    |-- kmsy
                    |-- ksea
                    |-- ksfo
                    |-- panc
```

### Installation

Make sure that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

**Recommended:** Use the  [`install.sh`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/install.sh) to download and install the Amelia Framework:

```bash
chmod +x install.sh
./install.sh amelia
```

This will create a conda environment named `amelia` and install all dependencies.

Alternatively, refer to [`INSTALL.md`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/INSTALL.md) for manual installation.

**Note:** AmeliaScenes only requires the Amelia dataset to run, only refer to AmeliaScenes installation.

## How to use

Activate your amelia environment (**Please follow the installation instructions above**):

```bash
conda activate amelia
```

### Generating scenes from raw data

Once you've installed the tools, and created the amelia environment, to process the data and generate the '.pkl' files, run:

```bash
cd amelia_scenes
python amelia_scenes/run_processor.py presets=amelia_10
```

This script will process all the raw trajectory CSV files found in `datasets/amelia/traj_data_a10v08/raw_trajectories/<airport_icao>/` and generate scenes for each the airports `kbos`, `kdca`, `kewr`, `kjfk`, `klax`, `kmdw`, `kmsy`, `ksea`, `ksfo` and `panc`.

where:

- `<airport_icao>`: [ICAO](https://en.wikipedia.org/wiki/ICAO_airport_code) code of the airport to be processed. It can be one of the following: `kbos`, `kdca`, `kewr`, `kjfk`, `klax`, `kmdw`, `kmsy`, `ksea`, `ksfo`, `panc`. By default it is set to `kbos`.
- `<parallel>`: If the processing should be done in parallel. By default it is set to `True`.

Since the script utilizes python hydra, additional parameters can also be changed on the configuration file found in `amelia_scenes/configs/processor/amelia_10.yaml`. For example, to change the percentage of data to be processed, the version of the trajectory data, or the number of parallel jobs, the script can be changed. As a recommendation allways process the data in parallel, unless debugging.

- `<to_process>`: What to process. By default is set to `both`. Possible options are:
  - `scenes`: only generate scenes from the raw files
  - `metas`: generates meta information from already generated scenes. It uses scene scoring tools.
  - `both`: generates scenes and meta information, simultaneously.
- `<base_dir>`: Path to the dataset. By default the path is set to `../datasets/amelia`.
- `<traj_version>`: Version of the trajectory data. By default it is set to `a10v08`.
- `<graph_version>`: Version of the graph data. By default it is set to `a10v01os`.
- `<overwrite>`: If the processing should overwrite the existing data. By default it is set to `True`.
- `<perc_process>`: Top limit visualization of the data being processed. By default it is set to `1.0`.
- `<seed>`: Seed for the random number generator. By default it is set to `42`.
- `<jobs>`: Number of Python worker processes to be used in parallel. By default it is set to `-1`, which will use all available CPUs.

#### Expected output

The scene processor should generate scene files for a given CSV file into ```datasets/amelia/traj_data_{version}/proc_trajectories/{airport_icao}/{raw_file_tagname}```. Each scene file is a pickle file following the format ```scene_id.pkl```.

For example if the input file is ```KBOS_1_1672531200.csv```, found in:

```bash
|-- datasets
    |-- amelia
        |-- traj_data_a10v08
            | -- raw_trajectories
                | -- kbos
                    | -- KBOS_1_1672531200.csv
                | -- other airports
```

The output scenes will be in:

 ```bash
|-- datasets
    |-- amelia
        |-- traj_data_a10v08
            | -- raw_trajectories
            | -- proc_scenes
                | -- kbos
                    | -- KBOS_1_1672531200
                        | -- 00000.pkl
                        ...
                        | -- xxxxx.pkl
```

---

### Characterizing scenes

**TODO: Work in Progress**

---

### Creating dataset splits

Once the scenes are generated, the `run_create_splits.py` script can be run to split the dataset. The script can be run as follows:

``` bash
cd amelia_scenes
python run_create_splits.py --split_type <random | day | month> --airport <airport_icao>
```

Where:

- `<split_type>`: Type of split to be generated. By default it is set to `random`. Possible options are:
  - `random`: Randomly splits the dataset into `train/val/test` sets.
  - `day`: Daily splits the dataset into `train/val/test` sets.
  - `month`: Monthly splits the dataset into `train/val/test` sets.
- `<airport_icao>`: ICAO code of the airport. By default it is set to `all`.

Additional parameters can also be specified:

``` bash
cd amelia_scenes
python run_create_splits.py --split_type <random | day | month> \
                        --base_dir <path_to_dataset> \
                        --seed <seed> \
                        --traj_version <version> \
                        --airport <airport_icao>
```

- `<base_dir>`: Path to the dataset. By default the path is set to `../datasets/amelia`.
- `<traj_version>`: Version of the trajectory data. By default it is set to `a10v08` to match the current released version.
- `<seed>`: Seed for the random number generator. By default it is set to `42`.
- `<airport>`: ICAO code of the airport. By default it is set to `all`.

#### Expected Output

For the `kbos` generated scene and the argument `--split_type` as `random`, the script should generate the following files:

```bash
|-- datasets
    |-- amelia
        |-- traj_data_a10v08
            |-- raw_trajectories
            |-- proc_scenes
            |-- splits
                |-- train_splits
                    |-- kbos_random.txt
                |-- val_splits
                    |-- kbos_random.txt
                |-- test_splits
                    |-- kbos_random.txt
```

## BibTeX

If you find our work useful in your research, please cite us!

```bibtex
@inbook{navarro2024amelia,
  author = {Ingrid Navarro and Pablo Ortega and Jay Patrikar and Haichuan Wang and Zelin Ye and Jong Hoon Park and Jean Oh and Sebastian Scherer},
  title = {AmeliaTF: A Large Model and Dataset for Airport Surface Movement Forecasting},
  booktitle = {AIAA AVIATION FORUM AND ASCEND 2024},
  chapter = {},
  pages = {},
  doi = {10.2514/6.2024-4251},
  URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2024-4251},
  eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2024-4251},
}
```
