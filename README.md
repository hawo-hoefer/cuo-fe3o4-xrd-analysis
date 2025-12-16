# HiVE ComaNet training repository

## Data generation
Our data generation program is available [here(TBD)](https://github.com/hawo-hoefer/yaxs). Follow the installation instructions there and install yaxs version 0.0.1.

### Quickstart for yaxs
Installation of yaxs requires a working rust toolchain. 
Install that from wherever you like (probably from your OS's repositories or using rustup).
Then:
```commandline
git clone https://github.com/hawo-hoefer/yaxs.git 
cd yaxs
cargo install --path .
```

### Computing the datasets
Simply generate the datasets using
```comandline
./generate.sh
```

Note that this requires some space. It will produce all 5 (noiseless) datasets mentioned in the paper.
These are 2 million XRD patterns with 2048 steps plus metadata each, resulting in approximately 80GB of data.

## Running the trainings and study
First, install all the training dependencies
```commandline
python3 -m virtualenv .venv
pip install -r requirements.txt
source .venv/bin/activate
```

Then, run the study using
```commandline
python3 ./train.py
```

