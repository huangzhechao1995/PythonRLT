cmake_minimum_required(VERSION 2.8.3)
project(RLT)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/RLT.h 
    ${CMAKE_CURRENT_SOURCE_DIR}/RegForestFit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/regForest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Utility/Utility.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Trees/Trees.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Trees/Miscellaneous.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Trees/Uni_Find_Terminal_Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utility/Utility.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_Split_A_Node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_Find_A_Split_Embed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_Find_A_Split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_Forest_Pred.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_Forest_Build.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_One_Split_Cat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Reg_Uni/Reg_Uni_One_Split_Cont.cpp
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