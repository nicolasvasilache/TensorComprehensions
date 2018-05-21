include_directories(.)
include_directories(..)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Examples
################################################################################
set(EXAMPLES_FILES
  batchmatmul
  group_convolution
  tmm
  MLP_model
)
foreach(i ${EXAMPLES_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(
     ${i}

     ${CUDA_CUBLAS_LIBRARIES}
     ${CUDA_CUDNN_LIBRARIES}
     ${GTEST_LIBS}
     ${GFLAGS_LIBRARIES}
     ${GLOG_LIBRARIES}

     ${ATEN_LIBRARIES}

     tc_caffe2_test_harness
     tc_autotuner
     tc_core
     tc_c2
     tc_aten
  )
endforeach()
