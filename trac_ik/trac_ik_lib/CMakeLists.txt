cmake_minimum_required(VERSION 2.8.3)
project(trac_ik_lib)

find_package(catkin REQUIRED
  COMPONENTS
    cmake_modules
    kdl_parser
    roscpp
    urdf
)

find_package(Boost REQUIRED COMPONENTS date_time)

find_package(PkgConfig REQUIRED)
pkg_check_modules(pkg_nlopt REQUIRED nlopt)
pkg_check_modules(Eigen REQUIRED eigen3)

catkin_package(
  CATKIN_DEPENDS
    kdl_parser
    roscpp
    urdf
  DEPENDS
    Boost
    pkg_nlopt
  INCLUDE_DIRS
    include
  LIBRARIES
    trac_ik
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${pkg_nlopt_INCLUDE_DIRS}
)

add_library(trac_ik
  src/kdl_tl.cpp
  src/nlopt_ik.cpp
  src/trac_ik.cpp)
target_link_libraries(trac_ik
  ${catkin_LIBRARIES}
  ${pkg_nlopt_LIBRARIES}
  ${Boost_LIBRARIES})

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

install(TARGETS trac_ik
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
