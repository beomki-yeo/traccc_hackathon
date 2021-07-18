# traccc

Demonstrator tracking chain for accelerators.

## Requirements and dependencies 

#### The following dependencies are needed
- `googletest` and `googletest/benchmark` are used as submodules for unit testing and benchmarking
- `dfelibs` is used as a submodule for `csv` reading/writing infrastructure
- `ACTS` is used as an external library

#### Data directory
- the `data` directory is a submodule hosted as `git lfs` on `https://gitlab.cern.ch/acts/traccc-data`

## Getting started

Clone the repository and setup up the submodules, this requires `git-lfs` for the data from the `traccc-data` repository.

```sh
git clone git@github.com:acts-project/traccc.git
cd traccc
git submodule update --init
```

### Build the project

```sh
cmake -S . -B <build_directory>
cmake --build <build_directory> <options>
```

Options | Default | Explanation |
--- | --- | --- |
TRACCC_BUILD_CUDA | OFF | Compile cuda directory |
Acts_DIR |  | Assign the ACTS cmake directory Ex) `THE_ACTS_INST_DIR/lib64/cmake/Acts` |
