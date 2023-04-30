cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(DAS_PROC LANGUAGES CXX)

#Required packages
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
add_subdirectory(external/pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall --std=c++11 -O3 -fPIC" )
set( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -O3 -fPIC" )

# Compiling source codes for processing
add_library(FILTERS proc_src/src/filters.cpp)

# Compiling pybind11 module
include_directories(proc_src/include)
pybind11_add_module(pyDAS MODULE proc_src/src/pyDAS.cpp)
target_link_libraries(pyDAS PRIVATE FILTERS ${PYTHON_LIBRARIES})
target_include_directories(pyDAS PRIVATE ${PYTHON_INCLUDE_DIRS})


# Installing library
set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install(TARGETS FILTERS DESTINATION lib) # C++ filtering library
install(TARGETS pyDAS DESTINATION ${PYTHON_DIR}) # Module for C++/Python processing
# install(FILES nonlinearPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
