cmake_minimum_required(VERSION 2.8.3)
project(trac_ik_kinematics_plugin)

find_package(PkgConfig REQUIRED)
pkg_check_modules(pkg_nlopt REQUIRED nlopt)

find_package(catkin REQUIRED
  COMPONENTS
    moveit_core
    pluginlib
    roscpp
    tf_conversions
    trac_ik_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${pkg_nlopt_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    moveit_core
    pluginlib
    roscpp
    tf_conversions
    trac_ik_lib
)

set(TRAC_IK_LIBRARY_NAME trac_ik_kinematics_plugin)

add_library(${TRAC_IK_LIBRARY_NAME} src/trac_ik_kinematics_plugin.cpp)
target_link_libraries(${TRAC_IK_LIBRARY_NAME} ${catkin_LIBRARIES} ${pkg_nlopt_LIBRARIES})

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

install(TARGETS ${TRAC_IK_LIBRARY_NAME}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(
  FILES
    trac_ik_kinematics_description.xml
  DESTINATION
    ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
