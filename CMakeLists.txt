cmake_minimum_required(VERSION 3.16)
project(IPC CXX)

set(CMAKE_CXX_STANDARD 17)

# Find the installed PolyFEM package
# The IPC library is included with PolyFEM
add_subdirectory(C:/Users/dsdsx/Code/libraries/polyfem polyfem)
add_subdirectory(CLI11)

add_executable(run_ipc main.cpp)

# Link against both PolyFEM and the IPC library
target_link_libraries(run_ipc PRIVATE polyfem::polyfem CLI11::CLI11)