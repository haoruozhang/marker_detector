cmake_minimum_required(VERSION 2.8.3)
project(marker_detector)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  cv_bridge
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
)

find_package(OpenCV REQUIRED )
find_package(OpenGL REQUIRED )
find_package(GLUT REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})
include_directories("src/${PROJECT_NAME}/include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIR})
link_directories(${OpenGL_LIBRARY_DIR})
link_directories(${GLUT_LIBRARY_DIRS})

add_executable(marker_main_node src/marker_main_node.cpp src/MarkerDetector.cpp src/GeometryTypes.cpp src/Marker.cpp)
add_executable(marker_get_image src/marker_get_image.cpp)

target_link_libraries(marker_main_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${OPENGL_LIBRARIES}
   ${GLUT_LIBRARY}
 )
target_link_libraries(marker_get_image
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${OPENGL_LIBRARIES}
   ${GLUT_LIBRARY}
 )

