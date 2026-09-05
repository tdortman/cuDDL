# Put the pinned headers before any CCCL headers exported by CUDA::toolkit.
get_filename_component(cccl_source "${CCCL_DIR}/../../.." ABSOLUTE)
# Meson's CMake importer drops include_directories pointing at sibling subprojects.
add_compile_options(
  "-I${cccl_source}/libcudacxx/include"
  "-I${cccl_source}/cub"
  "-I${cccl_source}/thrust")

# Export the same precedence to Meson consumers after NVBench creates its targets.
cmake_language(DEFER CALL target_compile_options nvbench INTERFACE
  "-I${cccl_source}/libcudacxx/include"
  "-I${cccl_source}/cub"
  "-I${cccl_source}/thrust")
