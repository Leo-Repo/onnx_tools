cmake_minimum_required(VERSION 3.18)
project(ort_custom_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ORT_ROOT)
  message(FATAL_ERROR "Please set ORT_ROOT to your ONNX Runtime install root.")
endif()

set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_INCLUDE_DIR_ALT "${ORT_ROOT}/include")
if(WIN32)
  set(ORT_LIB "${ORT_ROOT}/lib/onnxruntime.lib")
else()
  set(ORT_LIB "${ORT_ROOT}/lib/libonnxruntime.so")
endif()

add_library(ort_custom_ops SHARED src/conv_relu6_custom_op.cc)
target_include_directories(ort_custom_ops PRIVATE "${ORT_INCLUDE_DIR}" "${ORT_INCLUDE_DIR_ALT}")
target_link_libraries(ort_custom_ops PRIVATE "${ORT_LIB}")

if(WIN32)
  target_compile_definitions(ort_custom_ops PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
