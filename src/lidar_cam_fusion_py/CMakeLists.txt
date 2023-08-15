cmake_minimum_required(VERSION 3.0.2)
project(lidar_cam_fusion_py)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

add_definitions(-DEIGEN_NO_DEBUG)
add_definitions(-DEIGEN_MPL2_ONLY)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# find_package(catkin REQUIRED COMPONENTS
#   roslaunch
#   rospy
#   sensor_msgs
#   std_msgs
#   message_filters
#   roscpp
#   pcl_conversions
#   pcl_ros
#   autoware_perception_msgs
#   cv_bridge
# )

find_package(catkin REQUIRED COMPONENTS
  roslaunch
  rospy
  sensor_msgs
  std_msgs
  message_filters
  roscpp
  pcl_conversions
  pcl_ros
  cv_bridge
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(BLAS REQUIRED)
# find_package(OpenMP REQUIRED)

# if(OPENMP_FOUND)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  # CATKIN_DEPENDS roscpp sensor_msgs message_filters pcl_conversions pcl_ros autoware_perception_msgs
  CATKIN_DEPENDS roscpp sensor_msgs message_filters pcl_conversions pcl_ros munkres
  DEPENDS OpenCV
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
 

# add_executable(lidar_cam_fusion_py_node src/lidar_cam_fusion_py_node.py src/utils.py)
# add_dependencies(lidar_cam_fusion_py_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(lidar_cam_fusion_py_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${BLAS_LIBRARIES} ${PCL_LIBRARIES})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

find_package(rosinstall)
execute_process(COMMAND pip install --no-cache-dir -r ${PROJECT_SOURCE_DIR}/src/requirements.txt --ignore-installed --target=${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION})