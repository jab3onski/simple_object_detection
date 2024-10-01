cmake_minimum_required(VERSION 3.5.0)
project(prymat_detection)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(prymat_detection main.cpp)

target_link_libraries( prymat_detection ${OpenCV_LIBS} )
set_property(TARGET prymat_detection PROPERTY CXX_STANDARD 20)