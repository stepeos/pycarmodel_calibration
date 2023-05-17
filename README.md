![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/stepeos/294e13317bfd466118101ec156067757/raw) ![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/stepeos/00d7e970025d1bf6f150568dc326c50d/raw) ![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/stepeos/a9924565e021b9c897e7e89e3ea4163b/raw)

# Pycalibration_tool

## CLI usage
Use `calibration-tool --help` for help. The help updates the information displayed for different actions, e.g. `calibration-tool --action=calibration --help` will have different options.
```
usage: cmd.py Data-Directory Output-Directory differential_evolution [-h] [--num-workers NUM_WORKERS] [--population-size POPULATION_SIZE] [--max-iter MAX_ITER] [--models MODELS]
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
calibration-tool --action=calibrate "path/to/input/data" "path/to/output-data/directory" differential_evolution --population-size=200 --max-iter=100 --param-keys="speedFactor,minGap,accel,decel,startupDelay,tau,delta,tpreview,tPersDrive,tPersEstimate,treaction,ccoolness,jerkmax,epsilonacc,taccmax,Mflatness,Mbegin"
```

### Installation
#### Prerequisites
1. Follow the SUMO installation instructions: https://sumo.dlr.de/docs/Installing/index.html and remember the installation directory or copy it to clipboard
2. Open a command-prompt, and verify that sumo and netgenerate are in the PATH variable:
   ```
   sumo --version
   netgenerate
   ```
3. If both commands show their info, then skip to step 4
   If not, add /path/to/sumo/bin to your `PATH` environment variable, this is platform specific.
4. Add `SUMO_HOME` environment variable by adding `SUMO_HOME=path/to/sumo`. This is platform specific.

#### Installation by building from source
1. Ensure that `python --version` returns `3.8.*`. If not, install anaconda and create a python 3.8.* environment or install python directly without anaconda (not recommended).
   ```
   $ python --version
   Python 3.8.16
   ```
2. ```
    pip install setuptools
    git clone https://github.com/stepeos/pycalibration_tool.git
    cd pycalibration_tool
    python3 -m pip install --upgrade build
    python3 -m build
    pip install ./dist/calibration_tool-0.1-py3-none-any.whl
    ```
3. successful, if `calibration-tool --help` shows the help