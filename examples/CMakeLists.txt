add_executable(debug debug/main.c)

target_link_libraries(debug mlp)

add_executable(emnist emnist/main.c)

target_link_libraries(emnist mlp)