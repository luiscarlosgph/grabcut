# Grabcut library
cuda_add_library(grabcut SHARED grabcut.cpp gmm.cu util.cu)
target_link_libraries(grabcut
  ${Python3_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pbcvt}
)

# Grabcut binary
cuda_add_executable(MainGrabcut main_grabcut.cpp) 
target_link_libraries(MainGrabcut 
  grabcut
  ${Python3_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pbcvt}
)

# Install libgrabcut.so (by default in /usr/local/lib)
install(TARGETS grabcut LIBRARY DESTINATION lib)

# Create dynamic link inside the folder of Python packages so that the library can be imported
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
${CMAKE_INSTALL_PREFIX}/lib/libgrabcut.so
${Python3_SITELIB}/grabcut.so
)")
