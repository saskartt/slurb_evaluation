PALM-SLUrb model experiments
==============================

Tools for performing sensitivity testing and evaluation for the PALM-SLUrb model.

The job generation and analysis is performed using Jupyter notebooks located at `notebooks/`, with longer parts of the code imported from separate source files located at `src/` to keep the notebooks tidy. The notebooks provide a rough overview of the processing and analysis workflow.

As the simulation output data is provided with the manuscript, it is not necessary to run the job generation again to reproduce the analysis. However, the job generation scripts are provided so that anyone could reproduce the simulations if they wish so. The job generation, depending on the system, might take even multiple hours to complete as hundreds of gigabytes of turbulence inflow data needs to be processed. Please note, that re-running the simulations yourself probably won't reproduce the results bit-perfectly due to differences in compiler optimizations, computing environment, etc. However, any differences should be insignificant in the context of the analysis and evaluation presented in the manuscript.

There are some global settings that can be changed in `config.yml`.

You can utilize the `requirements.txt` or `environment.yml` files to create the required Python environment.

# Data
The input and output data of the simulations are provided at ... The data needs to be placed in the `data/` directory of the project root. As the complete dataset is relatively large, it might be necessary to link `data` to a directory on a larger external storage, e.g. `ln -s /path/to/data data`.

# Generating PALM jobs
The job generation is split into two phases as the simulation outputs from the precursor runs are required to produce the turbulent inflow and other boundary conditions for the main simulation runs. The job files will be placed into `data/jobs/`.

Note that the job generation handles large data files (>100 GiB), such as the turbulent inflow data. Dask arrays are used to split the processing into smaller chunks that fit the memory and that can be executed in parallel. The chunk size can be controlled in `config.yml`. The processing has been tested with 32 GiB and 128 GiB of RAM. However, I cannot promise that the processing will run on your system.

## How is the job generation controlled?
Some parts of the job generation is done manually, some automatically based on configuration files. The configuration files for jobs are located at `experiments/`. These include the base namelists as YAML configuration files, which are easier to handle and write than Fortran namelists. The scripts will automatically format and export the namelists (`p3d` and `p3dr`) for the jobs based on these configurations. Some of the values in the namelist configurations are set to `null`, these are determined and set runtime in the scripts based on other data sources.

Furthermore, the generation of the sensitivity tests is highly automated. The experiments are described in a YAML configuration file `experiments/sensitivity/experiments.yml`. The scripts go through this configuration and gernerate the sensitivity tests by applying modifications to the baseline case accordingly. The automatic generation minimises the risk of human errors when generation tens (>70) of PALM jobs, and, of course saves a lot of time by minimising manual labour.

Note that the model comparison setups have some manual adjustments in the namelist files considering the outputs. The final namelists are provided along with the rest of the simulation data.

# Author
Sasu Karttunen \
<sasu.karttunen@helsinki.fi>

# Copyright
(c) University of Helsinki, 2024 \
Licensed under the EUPL