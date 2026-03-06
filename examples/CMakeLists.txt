file(GLOB EXAMPLE_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

foreach(EXAMPLE ${EXAMPLE_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE})
        file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}/*.c
        )

        add_executable(${EXAMPLE} ${EXAMPLE_SOURCES})

        target_link_libraries(${EXAMPLE} mlp dataset)
    endif()
endforeach()
