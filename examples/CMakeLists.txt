file(GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(EXAMPLE ${EXAMPLE_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE})
        add_executable(${EXAMPLE} ${EXAMPLE}/main.c)
        target_link_libraries(${EXAMPLE} mlp)
    endif()
endforeach()
