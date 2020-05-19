cmake_minimum_required(VERSION 3.4.3)

project(test_ie CXX)

find_package(InferenceEngine REQUIRED)
# find_package(ngraph REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_range_for)

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${InferenceEngine_LIBRARIES}
#   ${NGRAPH_LIBRARIES}
)
