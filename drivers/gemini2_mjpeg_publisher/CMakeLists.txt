cmake_minimum_required(VERSION 3.8)
project(gemini2_mjpeg_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

add_executable(gemini2_mjpeg_node src/gemini2_mjpeg_node.cpp)
target_compile_features(gemini2_mjpeg_node PUBLIC cxx_std_17)
target_include_directories(gemini2_mjpeg_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gemini2_mjpeg_node ${OpenCV_LIBRARIES})
ament_target_dependencies(gemini2_mjpeg_node rclcpp sensor_msgs)
install(TARGETS gemini2_mjpeg_node DESTINATION lib/${PROJECT_NAME})
ament_package()

