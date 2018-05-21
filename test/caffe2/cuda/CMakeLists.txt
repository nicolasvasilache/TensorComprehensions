include_directories(.)
include_directories(..)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# Caffe2 + TC tests
################################################################################
set(TEST_FILES
  test_caffe2
)

foreach(i ${TEST_FILES})
  add_executable(${i} ${i}.cc)
  add_test(${i} ${i})
  target_link_libraries(${i} ${GOOGLE_LIBS} pthread tc_caffe2_test_harness tc_c2)
endforeach()
