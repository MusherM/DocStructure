cmake_minimum_required(VERSION 3.4.1)
project(UniRecDynamicOm)

find_library(CANN_LIB hiai_foundation)

add_library(entry SHARED
  diagnostic_log.cpp
  napi_init.cpp
  om_model.cpp
  tokenizer.cpp
  unirec_engine.cpp
)

target_include_directories(entry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(entry PUBLIC
  libace_napi.z.so
  libhilog_ndk.z.so
  librawfile.z.so
  ${CANN_LIB}
  libneural_network_core.so
)
