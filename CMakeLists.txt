cmake_minimum_required(VERSION 3.0.2)
project(kinova_gen3_teleop)

find_package(catkin REQUIRED COMPONENTS
  moveit_commander
  moveit_msgs
  moveit_ros_planning_interface
  kortex_driver
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/leap_teleop.py
  scripts/leap_to_kinova_dual.py
  scripts/teleop_dual_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES scripts/leap_common.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
