#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
./build/src/test_radhydro3d_shell tests/radhydro_shell_amr.in amrex.async_out=1