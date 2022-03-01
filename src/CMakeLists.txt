cmake_minimum_required(VERSION 2.8.3)
project(RLT)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/RLT.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/RegForestFit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/regForest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Utility/Utility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Utility/Utility.cpp 
    )

find_package(Armadillo REQUIRED)

add_executable(
    rlt
    ${SOURCES}
    )

target_include_directories(
    rlt
    PUBLIC
    ${ARMADILLO_INCLUDE_DIR}
)


target_link_libraries(rlt ${ARMADILLO_LIBRARIES})