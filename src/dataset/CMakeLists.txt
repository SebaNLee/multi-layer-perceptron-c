file(GLOB DATASET_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

add_library(dataset ${DATASET_SOURCE})

target_include_directories(dataset PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_compile_options(dataset PRIVATE -Wall -pedantic)
