function(target_resources _target)
  set(_kwargs PUBLIC PRIVATE INTERFACE)
  cmake_parse_arguments(ARG
      ""
      ""
      "${_kwargs}"
      ${ARGN})

  foreach(_section IN LISTS _kwargs)
    set(_resource_paths)
    foreach(_res IN LISTS ARG_${_section})
      # Compute path of file relative to the current source dir
      get_filename_component(_res_full_path ${_res} ABSOLUTE)
      file(RELATIVE_PATH _res_path ${CMAKE_CURRENT_SOURCE_DIR} ${_res_full_path})

      # Translate this to a path in the current binary dir
      set(_absolute_bin_path ${CMAKE_CURRENT_BINARY_DIR}/${_res_path})
      list(APPEND _resource_paths ${_absolute_bin_path})

      # Add a generator for this file that copies the file, this won't itself add any dependency
      # between _target and the file
      add_custom_command(OUTPUT ${_res_path}
          COMMAND ${CMAKE_COMMAND} -E copy "${_res_full_path}" "${_absolute_bin_path}"
          )
    endforeach()

    # Adding the generated file as a source will create the required dependency
    target_sources(${_target} ${_section} ${_resource_paths})
  endforeach()
endfunction()
