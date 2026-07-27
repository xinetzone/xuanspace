set({{module_name|upper}}_SOURCES
  ffi_registry.cc
)

if({{package_name|upper}}_USE_STUB)
  message(STATUS "Using stub {{module_name}} runtime implementation")
  list(APPEND {{module_name|upper}}_SOURCES stub_rt.cc)
else()
  message(STATUS "Using real {{module_name}} runtime implementation")
  list(APPEND {{module_name|upper}}_SOURCES real_rt.cc)
endif()

add_library({{package_name}}_{{module_name}} SHARED ${{{module_name|upper}}_SOURCES})

target_include_directories({{package_name}}_{{module_name}} PUBLIC ${{{package_name|upper}}_INCLUDE_DIR})

if(NOT {{package_name|upper}}_USE_STUB)
  target_link_libraries({{package_name}}_{{module_name}} PRIVATE {{module_name|upper}}::{{module_name|upper}})
endif()

tvm_ffi_configure_target({{package_name}}_{{module_name}}
  STUB_DIR "${CMAKE_SOURCE_DIR}/python"
  STUB_INIT OFF
)

install(TARGETS {{package_name}}_{{module_name}} DESTINATION .)
tvm_ffi_install({{package_name}}_{{module_name}} DESTINATION .)
