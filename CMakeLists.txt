cmake_minimum_required(VERSION 3.5)
project(orbslam3)

set(ENV{PYTHONPATH} "/opt/ros/jazzy/lib/python3.12/site-packages/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)
set(ONNXRUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo/onnxruntime-linux-x64-1.20.1)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(ORB_SLAM3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
  include
  ${ORB_SLAM3_ROOT_DIR}/include
  ${ORB_SLAM3_ROOT_DIR}/include/CameraModels
  ${ONNXRUNTIME_DIR}/include
)

link_directories(include)

add_executable(mono
  src/monocular/mono.cpp
  src/monocular/monocular-slam-node.cpp
  src/yolo/src/image_inference.cpp
  src/yolo/src/YOLO11.cpp
)
target_link_libraries(mono ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so yaml-cpp)

ament_target_dependencies(mono rclcpp sensor_msgs cv_bridge ORB_SLAM3 Pangolin OpenCV tf2_ros ament_index_cpp visualization_msgs geometry_msgs nav_msgs)

install(TARGETS mono
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config vocabulary rviz
 DESTINATION share/${PROJECT_NAME}/)

install(DIRECTORY launch config vocabulary
 DESTINATION share/${PROJECT_NAME}/)

install(
  DIRECTORY src/yolo/models
  DESTINATION share/${PROJECT_NAME}/
)
install(
  DIRECTORY  ${ONNXRUNTIME_DIR}/lib/
  DESTINATION lib
)

install(
  DIRECTORY
  DESTINATION share/${PROJECT_NAME}/result/img
)

ament_package()

