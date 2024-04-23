![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/stepeos/294e13317bfd466118101ec156067757/raw) ![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/stepeos/00d7e970025d1bf6f150568dc326c50d/raw) ![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/stepeos/a9924565e021b9c897e7e89e3ea4163b/raw)

# Pycarmodel_calibration
This carmodel calibration tool was developed for my student thesis. The purpose is to calibrate the Extended Intelligent Driver Model (EIDM) from dron data. The private dataset contains drone footage of vehicles driving off at an intersection, similiar to [this video](https://www.youtube.com/watch?v=Vz4f8Gy6P1Q).
The calibration tool would first find a selection of vehicles in pairs, each one containing a leader and a follower. Using [SUMO](https://eclipse.dev/sumo/) to simulate the EIDM behavior, an optimization algorithm finds the best set of parameters from the dataset's trajectory. The leader's trajectory is the input and the follower's trajectory is simulated. The basis if calibration is the deviation between the follower's simulated trajectory to the real follower's trajectory.


## CLI usage
Use `carmodel-calibration --help` for help. The help updates the information displayed for different actions, e.g. `carmodel-calibration --action=calibration --help` will have different options.
```
usage: carmodel-calibration Data-Directory Output-Directory differential_evolution [-h] [--num-workers NUM_WORKERS] [--population-size POPULATION_SIZE] [--max-iter MAX_ITER] [--models MODELS]
                                                                     [--param-keys PARAM_KEYS] [--force-selection] [--no-report]

optional arguments:
  -h, --help            show this help message and exit
  --population-size POPULATION_SIZE
                        Size of the populationdoes not apply to direct.
  --max-iter MAX_ITER   Number of max iterations.
  --models MODELS       Models to train as comma separated list e.g. `eidm`.
  --param-keys PARAM_KEYS
                        comma separated list of parameters e.g. `speedFactor,minGap,...`
  --force-selection     If the flag is passed, then the selection is forced to be recalculateld.
  --no-report           Will not create reports in the output directory.
```


Here's an example usage using differential evolution as calibration algorithm.
```
carmodel-calibration --action=calibrate "path/to/input/data" "path/to/output-data/directory" differential_evolution --population-size=200 --max-iter=100 --param-keys="speedFactor,minGap,accel,decel,startupDelay,tau,delta,tpreview,tPersDrive,tPersEstimate,treaction,ccoolness,jerkmax,epsilonacc,taccmax,Mflatness,Mbegin"
```

## Installation
### Prerequisites
1. Follow the SUMO installation instructions: https://sumo.dlr.de/docs/Installing/index.html and remember the installation directory or copy it to clipboard
2. Open a command-prompt, and verify that sumo and netgenerate are in the PATH variable:
   ```
   sumo --version
   netgenerate
   ```
3. If both commands show their info, then skip to step 4
   If not, add /path/to/sumo/bin to your `PATH` environment variable, this is platform specific.
4. Add `SUMO_HOME` environment variable by adding `SUMO_HOME=path/to/sumo`. This is platform specific.

### Installation by building from source
1. Ensure that `python --version` returns `3.8.*`. If not, install anaconda and create a python 3.8.* environment or install python directly without anaconda (not recommended).
   ```
   $ python --version
   Python 3.8.16
   ```
2. ```
    pip install setuptools
    git clone https://github.com/stepeos/pycarmodel_calibration.git
    cd pycarmodel_calibration
    python3 -m pip install --upgrade build
    python3 -m build
    pip install ./dist/carmodel_calibration-0.1-py3-none-any.whl
    ```
3. successful, if `carmodel-calibration --help` shows the help

##  Example Usage:
For educational purposes, InD dataset can be acquired for free from [levelxdata.com/ind-dataset/](https://levelxdata.com/ind-dataset/). This dataset does not contain any intersection with traffic lights, however with the right configuration file, some calibration targets can still be found.
1. Download the dataset to `/my/dataset/path/inD-dataset-v1.0/`
2. create a results directory `mkdir -p /my/dataset/path/inD-dataset-v1.0/results`
3. download the sample config `wget -O /my/dataset/path/inD-dataset-v1.0/data/data_config.json https://github.com/stepeos/pycarmodel_calibration/main/ind_config.json`
4. start the calibration process `carmodel-calibration --action=calibrate /my/dataset/path/inD-dataset-v1.0/data/ /my/dataset/path/inD-dataset-v1.0/results/ differential_evolution --population-size=200 --max-iter=100 --param-keys=speedFactor,minGap,accel,decel,startupDelay,tau,delta,tpreview,tPersDrive,tPersEstimate,treaction,ccoolness,jerkmax,epsilonacc,taccmax,Mflatness,Mbegin` (if the path contains spaces, one can use quotes like "/my-path/with whitespaces/inD-dataset-v1.0")


