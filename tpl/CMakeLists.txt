target_include_directories(resilience PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:include>
                           )

install(DIRECTORY pico DESTINATION include FILES_MATCHING PATTERN "*.h")
