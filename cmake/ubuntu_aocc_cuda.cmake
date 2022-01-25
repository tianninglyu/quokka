# host configuration
# run with `cmake -C host_config.cmake ..` from inside build directory

set(CMAKE_C_COMPILER "/opt/AMD/aocc-compiler-3.1.0/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/AMD/aocc-compiler-3.1.0/bin/clang++" CACHE PATH "")
set(AMReX_GPU_BACKEND CUDA CACHE STRING "")
set(AMReX_DIFFERENT_COMPILER ON CACHE BOOL "")

