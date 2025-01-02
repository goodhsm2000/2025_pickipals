# CMake generated Testfile for 
# Source directory: /home/turtle/catkin_ws/src/YDLidar-SDK-master/python
# Build directory: /home/turtle/catkin_ws/src/YDLidar-SDK-master/build/python
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ydlidar_py_test "/usr/bin/python" "/home/turtle/catkin_ws/src/YDLidar-SDK-master/python/test/pytest.py")
set_tests_properties(ydlidar_py_test PROPERTIES  ENVIRONMENT "PYTHONPATH=/home/turtle/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages:/home/turtle/catkin_ws/src/YDLidar-SDK-master/build/python" _BACKTRACE_TRIPLES "/home/turtle/catkin_ws/src/YDLidar-SDK-master/python/CMakeLists.txt;42;add_test;/home/turtle/catkin_ws/src/YDLidar-SDK-master/python/CMakeLists.txt;0;")
subdirs("examples")
