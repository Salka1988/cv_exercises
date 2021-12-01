# OpenCV:
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})
IF(OpenCV_FOUND)
  MESSAGE(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
ELSE(OpenCV_FOUND)
  MESSAGE(STATUS "OpenCV_LIBS not found!")
ENDIF(OpenCV_FOUND)

project(cvtask1b)
file(GLOB SOURCES ${SOURCE_WILDCARDS})
include_directories(${CMAKE_SOURCE_DIR}/cgcvcommon)
add_executable(cvtask1b ${SOURCES})
target_link_libraries(cvtask1b ${OpenCV_LIBS})
