cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(GMM)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

find_package(Eigen3 REQUIRED)

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
       "Choosing Release build type - the fastest option."
       FORCE )
endif()

include_directories(./ 
  ${EIGEN3_INCLUDE_DIRS})

add_library( GMM
  GMM.cpp
  Kmeans.cpp
  Matrix.cpp
)
add_executable( main
  main.cpp
)
target_link_libraries( main
  GMM
)

# export(TARGETS GMM
#   FILE "${PROJECT_BINARY_DIR}/GMMConfig.cmake")

export(PACKAGE GMM)

# file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
#    "${INSTALL_INCLUDE_DIR}")

# configure_file(GMMConfigVersion.cmake.in
#   "${PROJECT_BINARY_DIR}/GMMConfigVersion.cmake" @ONLY)

# install(FILES
#   "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/GMMConfig.cmake"
#   "${PROJECT_BINARY_DIR}/GMMConfigVersion.cmake"
#   DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)


# install(TARGETS GMM
#   # IMPORTANT: Add the bar executable to the "export-set"
#   EXPORT GMMTargets
#   RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

