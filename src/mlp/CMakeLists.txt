file(GLOB_RECURSE MLP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/*.c)

add_library(mlp ${MLP_SOURCE})

target_include_directories(mlp PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(mlp PUBLIC m)

target_compile_options(mlp PRIVATE -Wall -pedantic)
target_compile_options(mlp PRIVATE -fsanitize=address -g) # TODO debug
target_link_options(mlp PUBLIC -fsanitize=address -g) # TODO debug