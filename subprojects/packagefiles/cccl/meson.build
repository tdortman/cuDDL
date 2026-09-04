project('cccl', ['cpp', 'cuda'], version: '3.6.0')

cuda_runtime = dependency('CUDAToolkit', method: 'cmake', modules: ['CUDA::cudart', 'CUDA::cuda_driver'])

cccl_dep = declare_dependency(
    include_directories: include_directories('libcudacxx/include', 'cub', 'thrust'),
    # CMake's toolkit dependency also exports the system CCCL include directory.
    dependencies: cuda_runtime.partial_dependency(links: true, link_args: true),
)
meson.override_dependency('cccl', cccl_dep)
