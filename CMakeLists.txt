cmake_minimum_required(VERSION 3.21)

project(mlp)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(src/mlp)
# add_subdirectory(src/dataset) # TODO
add_subdirectory(examples)
