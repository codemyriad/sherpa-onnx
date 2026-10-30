include_directories(${PROJECT_SOURCE_DIR})
add_library(sherpa-onnx-c-api c-api.cc)
target_link_libraries(sherpa-onnx-c-api sherpa-onnx-core)
target_include_directories(sherpa-onnx-c-api PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_BUILD_MAIN_LIB=1)
endif()

add_library(sherpa-onnx-cxx-api cxx-api.cc)
target_link_libraries(sherpa-onnx-cxx-api sherpa-onnx-c-api)
target_include_directories(sherpa-onnx-cxx-api PUBLIC ${PROJECT_SOURCE_DIR})

# Cassini: Linux x86_64 shared CUDA builds also define and export
# cudaGetDeviceProperties(_v2), a per-device cache for the cuDNN-frontend that
# ONNX Runtime 1.28.2 bundles; see cassini-cuda-props-cache.cc and CASSINI.md.
set(SHERPA_ONNX_CASSINI_CUDA_PROPS_CACHE OFF)
if(SHERPA_ONNX_ENABLE_GPU AND BUILD_SHARED_LIBS
   AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(SHERPA_ONNX_CASSINI_CUDA_PROPS_CACHE ON)
  target_sources(sherpa-onnx-c-api PRIVATE cassini-cuda-props-cache.cc)
  target_link_libraries(sherpa-onnx-c-api ${CMAKE_DL_LIBS})
endif()
message(STATUS "SHERPA_ONNX_CASSINI_CUDA_PROPS_CACHE: ${SHERPA_ONNX_CASSINI_CUDA_PROPS_CACHE}")

if(SHERPA_ONNX_CASSINI_CUDA_PROPS_CACHE)
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c-cuda.lds"
  )
elseif(ANDROID OR (UNIX AND NOT APPLE))
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c.lds"
  )
elseif(APPLE)
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    LINK_FLAGS "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c.exp"
  )
endif()

install(
  TARGETS
    sherpa-onnx-c-api
    sherpa-onnx-cxx-api
  DESTINATION
    lib
)

install(
  FILES
    c-api.h
    cxx-api.h
  DESTINATION
    include/sherpa-onnx/c-api
)
