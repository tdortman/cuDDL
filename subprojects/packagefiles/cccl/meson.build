project('cccl', ['cpp', 'cuda'], version: '3.6.0', meson_version: '>= 1.1')

driver_dir = get_option('driver_library_dir')
cuda_runtime = dependency(
    'CUDAToolkit',
    method: 'cmake',
    modules: ['CUDA::cudart', 'CUDA::cuda_driver'],
    cmake_args: driver_dir == '' ? [] : ['-DCUDA_cuda_driver_LIBRARY=' + driver_dir / 'libcuda.so'],
)

cccl_dep = declare_dependency(
    include_directories: include_directories('libcudacxx/include', 'cub', 'thrust'),
    # CMake's toolkit dependency also exports the system CCCL include directory.
    dependencies: cuda_runtime.partial_dependency(links: true, link_args: true),
)
meson.override_dependency('cccl', cccl_dep)
