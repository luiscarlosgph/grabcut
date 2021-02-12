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
install(DIRECTORY ${CMAKE_SOURCE_DIR}/inc/grabcut DESTINATION include FILES_MATCHING PATTERN "*.h")

# Create dynamic link inside the folder of Python packages so that the library can be imported
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink 
${CMAKE_INSTALL_PREFIX}/lib/libgrabcut.so
${Python3_SITELIB}/grabcut.so
)")
