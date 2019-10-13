cmake_minimum_required (VERSION 3.0)
project (SpectralKDV)

# find python headers and lib files
find_package(PythonLibs 2.7 REQUIRED)

# include project headers
include_directories(include)

# include third party headers
include_directories(${CMAKE_SOURCE_DIR}/ThirdParty)
include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/matplotlib-cpp)

# add python headers
include_directories(${PYTHON_INCLUDE_DIRS})


# add executables
add_executable(mytest src/test2.C)
add_executable(ab2bd2_test testing/ab2bd2_tests.C)
add_executable(etdrk2_test testing/etdrk2_tests.C)

# link to python libs
target_include_directories(mytest PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(ab2bd2_test PRIVATE ${PYTHON_INCLUDE_DIRS})
message(STATUS ${PYTHON_INCLUDE_DIRS} " " ${PYTHON_LIBRARIES})
target_link_libraries(mytest fftw3 ${PYTHON_LIBRARIES})
target_link_libraries(ab2bd2_test fftw3 ${PYTHON_LIBRARIES})
target_link_libraries(etdrk2_test fftw3 ${PYTHON_LIBRARIES})
# install app
install(TARGETS mytest RUNTIME DESTINATION homework4)
install(TARGETS ab2bd2_test RUNTIME DESTINATION homework4)
