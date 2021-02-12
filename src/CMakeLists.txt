# Grabcut library
cuda_add_library(grabcut SHARED grabcut.cpp gmm.cu util.cu)
target_link_libraries(grabcut
  ${Python3_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)

# Grabcut binary
cuda_add_executable(main_grabcut main_grabcut.cpp) 
target_link_libraries(main_grabcut
  grabcut
  ${Python3_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)

# Install libgrabcut.so (by default in /usr/local/lib)
install(TARGETS grabcut LIBRARY DESTINATION lib)
install(FILES ../inc/grabcut/boost_graph.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/gmm.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/grabcut.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/graph.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/helper_cuda.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/helper_string.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/opencv_graph.h DESTINATION include/grabcut)
install(FILES ../inc/grabcut/util.h DESTINATION include/grabcut)

# Create dynamic link inside the folder of Python packages so that the library can be imported
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
${CMAKE_INSTALL_PREFIX}/lib/libgrabcut.so
${Python3_SITELIB}/grabcut.so
)")
