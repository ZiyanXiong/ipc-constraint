cmake_minimum_required(VERSION 3.14)
project(polyfem_mex_wrapper)

# 1. Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find MATLAB
find_package(Matlab REQUIRED)

add_subdirectory($ENV{POLYFEM_DIR} polyfem)

add_subdirectory(Exact-Root-Parity-CCD)

# =========================================================
# HELPER FUNCTION: Define a standard Polyfem MEX Function
# =========================================================
function(add_polyfem_mex target_name source_file)
    # A. Create the MEX target using the standard MATLAB CMake command
    matlab_add_mex(
        NAME ${target_name}
        SRC ${source_file}
    )

    # B. Link against Polyfem
    target_link_libraries(${target_name} polyfem::polyfem ccd::double)
endfunction()

# =========================================================
# REGISTER YOUR MEX FUNCTIONS HERE
# =========================================================

# 1. Your main IPC solver
add_polyfem_mex(polyfem_ipc_mex polyfem_ipc_mex.cpp)

# 2. Example: A second MEX function 
add_polyfem_mex(polyfem_ipc_step_mex polyfem_ipc_step_mex.cpp)

# 3. Example: Another function
add_polyfem_mex(polyfem_ipc_problem_mex polyfem_ipc_problem_mex.cpp)