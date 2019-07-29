cmake_minimum_required( VERSION 2.8 )
project (gpu_sift)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/gcc-6;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 

set(cuda_sources
  cudaImage.cu  
  cudaImage.h  
  cudaSiftH.cu 
  cudaSiftH.h  
  matching.cu  
  cudaSiftD.h  
  cudaSift.h  
  cudautils.h
)  

set(sources
  geomFuncs.cpp  
  mainSift.cpp  
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} )

cuda_add_library( sift_gpu SHARED 
		  cudaImage.cu  
		  cudaSiftH.cu 
		  matching.cu 
		  )
		  
target_link_libraries (sift_gpu 
		       ${CUDA_cudadevrt_LIBRARY} 
		       ${OpenCV_LIBS})

cuda_add_executable(cudasift ${cuda_sources} ${sources} OPTIONS -arch=sm_35)
target_link_libraries(cudasift ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})


