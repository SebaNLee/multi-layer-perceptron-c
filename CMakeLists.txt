cmake_minimum_required(VERSION 3.21)

project(mlp)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(WARNING "LTO/IPO not supported: ${IPO_ERROR}")
endif()

add_compile_options($<$<CONFIG:Release>:-march=native>)
add_compile_options( $<$<CONFIG:Release>:-mtune=native>)
add_compile_options($<$<CONFIG:Debug>:-fsanitize=address>)
add_compile_options($<$<CONFIG:Debug>:-fno-omit-frame-pointer>)
add_link_options($<$<CONFIG:Debug>:-fsanitize=address>)

add_subdirectory(src/mlp)
add_subdirectory(src/dataset)
add_subdirectory(examples)
