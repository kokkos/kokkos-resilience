#!/bin/bash

set -x

cmake --preset ci
cmake --build --preset ci
ctest --preset ci
