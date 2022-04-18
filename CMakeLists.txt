cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

add_subdirectory(pybind11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RLT.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RegForestFit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/regForest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utility/Utility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utility/Utility.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Trees/Trees.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Trees/Definition.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Trees/Miscellaneous.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Trees/Uni_Find_Terminal_Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_Split_A_Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_Find_A_Split_Embed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_Find_A_Split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_Forest_Pred.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_Forest_Build.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_One_Split_Cat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Reg_Uni/Reg_Uni_One_Split_Cont.cpp
    )

find_package(Armadillo REQUIRED)
message(STATUS "ARMADILLO_INCLUDE_DIR is ${ARMADILLO_INCLUDE_DIR}")
#add_executable(
#    cmake_example
#    ${SOURCES}
#    )
#target_include_directories(
#    cmake_example
#    PUBLIC
#    ${ARMADILLO_INCLUDE_DIR}
#)
#target_link_libraries(cmake_example ${ARMADILLO_LIBRARIES})

pybind11_add_module(cmake_example SHARED ${SOURCES})

#target_include_directories(
#    cmake_example
#    PUBLIC
#    ${ARMADILLO_INCLUDE_DIR}
#)
#target_link_libraries(cmake_example ${ARMADILLO_INCLUDE_DIR})


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
