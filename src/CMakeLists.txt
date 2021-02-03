# Grabcut library
cuda_add_library(grabcut SHARED grabcut.cpp gmm.cu util.cu)
target_link_libraries(grabcut
  opencv_core 
  opencv_highgui
  opencv_imgproc 
  opencv_imgcodecs 
  nppc 
  python2.7 
  ${BOOST_PYTHON_LIB_DIR}/libboost_python27.a 
  ${CMAKE_SOURCE_DIR}/pyboostcvconverter/build/libstatic_pbcvt.a
)
#set_target_properties(grabcut PROPERTIES PREFIX "")

# GMM binary
cuda_add_executable(MainGMM main_gmm.cpp gmm.cu util.cu)
target_link_libraries(MainGMM 
  opencv_core 
  opencv_highgui 
  opencv_imgproc 
  opencv_imgcodecs 
  nppc
)

# Grabcut binary
cuda_add_executable(MainGrabcut main_grabcut.cpp gmm.cu util.cu) 
target_link_libraries(MainGrabcut 
  opencv_core 
  opencv_highgui 
  opencv_imgproc 
  opencv_imgcodecs 
  nppc 
  python2.7 
  ${BOOST_PYTHON_LIB_DIR}/libboost_python27.a 
  ${CMAKE_SOURCE_DIR}/pyboostcvconverter/build/libstatic_pbcvt.a
)

# libgrabcut.so installation configuration
install(TARGETS grabcut LIBRARY DESTINATION lib) 
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
  ${SHARED_LIB_PATH} 
  ${SHARED_LIB_LINK}
)")
