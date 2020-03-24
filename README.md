# Kokkos Resilience

*Kokkos Resilience* is an extension to [*Kokkos*](https://github.com/kokkos/kokkos/) for providing convenient resilience
and checkpointing to scientific applications.

## Building

*Kokkos Resilience* is built using [CMake](https://cmake.org) version 3.14 or later. It has been tested on recent
compilers such as GCC 9.3.0 and LLVM/Clang 9.0.0. It should work on any C++14 supporting compiler, but your mileage
may vary.

### Dependencies

First and foremost, *Kokkos Resilience* requires an install of *Kokkos*. This can be compiled or a version bundled with
other software (such as Trilinos) or as a package on a machine.

**Note:** *Kokkos Resilience* currently requires some internal changes to Kokkos that are not yet merged in. Use the
[kokkos-fork](https://gitlab-ex.sandia.gov/kokkos-resilience/kokkos) repository (altdev branch) in the meantime.

Additionally, *Kokkos Resilience* uses the [Veloc](https://github.com/ECP-VeloC/VELOC) library for efficient asynchronous
checkpointing. If you desire automatic checkpointing to be available this library (and additionally MPI) must be installed.

#### Obtaining VeloC

We are maintaining a special spack package for VeloC since the main one is not up-to-date. It can be found
[here](https://gitlab-ex.sandia.gov/kokkos-resilience/kr-spack) and can be installed via:

```sh
git clone git@gitlab-ex.sandia.gov:kokkos-resilience/kr-spack.git
spack repo add kr-spack
spack install veloc@barebone
```

It is recommended to install the "barebone" variant/branch of VeloC since it has reduced dependencies.

### CMake Invocation

Typically, invoking CMake involves the creation of a build directory. From the build directory, invoke CMake as (for example) follows:

```sh
cmake -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ROOT=/path/to/kokkos/install/install \
      -DVeloC_ROOT=/path/to/veloc/install \
      -DVELOC_BAREBONE=ON \
      path/to/source/dir
```

For a more detailed summary of compiler switches please see below.

### CMake paths

| Path        | Description                                             |
| ----------- | ------------------------------------------------------- |
| Kokkos_ROOT | Path to the root of the Kokkos install                  |
| VeloC_ROOT  | Path to the root of VeloC if it is enabled (see below)  |
| HDF5_ROOT   | Path to the root of HDF5 if HDF5 is enabled (see below) |


### Supported CMake Options

| Variable                | Default | Description                                        |
| ----------------------- | ------- | -------------------------------------------------- |
| KR_ENABLE_VELOC         | ON      | Enables the VeloC backend                          |
| VELOC_BAREBONE          | OFF     | Enable VeloC barebone mode                         |
| KR_ENABLE_TRACING       | OFF     | Enable performance tracing of resilience functions |
| KR_ENABLE_STDIO         | OFF     | Use stdio for manual checkpoint                    |
| KR_ENABLE_HDF5          | OFF     | Add HDF5 support for manual checkpoint             |
| KR_ENABLE_HDF5_PARALLEL | OFF     | Use parallel version of HDF5 for manual checkpoint |
| KR_ENABLE_TESTS         | ON      | Enable tests in the build                          |
| KR_ENABLE_EXAMPLES      | ON      | Enable examples in the build                       |
