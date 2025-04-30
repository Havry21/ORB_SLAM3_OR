add_library(yolo STATIC)

set(ONNXRUNTIME_DIR onnxruntime-linux-x64-1.20.1)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)
include_directories("../include/")

target_sources(yolo src/image_inference.cpp)

target_include_directories(yolo PRIVATE "${ONNXRUNTIME_DIR}/include")
target_compile_features(yolo PRIVATE cxx_std_20)
target_link_libraries(yolo ${OpenCV_LIBS})
target_link_libraries(yolo ${CMAKE_CURRENT_SOURCE_DIR}/${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
message(${CMAKE_CURRENT_SOURCE_DIR}/${ONNXRUNTIME_DIR})
