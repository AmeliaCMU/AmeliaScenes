# Amelia Framework

## Install Script (recommended)

### Prerequisites

To install the Amelia framework using the provided bash script, make it executable and run, specifying the desired environment name. Make sure to have conda installed, as well as GitHub configured with a [SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) for cloning.

```bash
chmod +x ./install.sh
./install.sh ameliacmu
```

This will clone the necessary repos for using the Amelia framework and install them as Python Packages in a conda environment; as well as installing the requiered dependencies.

## Manual Installation

Installation can be done by installing the AmeliaScenes repository or the complete Amelia Framework

### Creating an environment

It can be created using either `conda`.

- Conda

```bash
conda create --name amelia python=3.9
conda activate amelia
```

### Installing Dependencies

#### AmeliaScenes

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaScenes.git
cd AmeliaScenes
pip install -e .
```

#### AmeliaMaps

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaMaps.git
cd AmeliaMaps
pip install -e .
```

#### AmeliaTF

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaTF.git
cd AmeliaTF
pip install -e .
```

**Note:** AmeliaTF requires AmeliaScenes dependencies to run.

#### AmeliaInference

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaInference.git
cd AmeliaInference
pip install -e .
```

**Note:** AmeliaInference requires AmeliaTF dependencies to run.

#### AmeliaDataTools

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaDataTools.git
cd AmeliaDataTools
pip install -e .
```

**Note:** AmeliaDataTools requires AmeliaScenes dependencies to run.

#### AmeliaSWIM

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaSWIM.git
cd AmeliaSWIM
pip install -e .
```
