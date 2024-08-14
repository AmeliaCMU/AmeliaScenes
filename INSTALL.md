# Amelia Framework

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

**Note:** AmeliaDataTools requires AmeliaScenes dependencies to run.
