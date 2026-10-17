project(
    'cuDDL',
    ['cuda', 'cpp'],
    version: '0.58.0',
    meson_version: '>= 1.3.0',
    default_options: [
        'warning_level=2',
        'cpp_std=c++20',
        'optimization=3',
    ],
)

cuda_arch_args = [
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_90,code=sm_90',
    '-gencode=arch=compute_80,code=sm_80',
]

cuda_args = [
    '-std=c++20',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '--generate-line-info',
] + cuda_arch_args

# CMake's CUDAToolkit falls back to the toolkit's stub libcuda/libnvidia-ml when the toolkit has
# no real driver, and Meson then records the stub directory as an RPATH that shadows the driver
# at run time. driver_library_dir names the real driver libraries instead.
driver_library_dir = get_option('driver_library_dir')
cuda_driver_cmake_defines = driver_library_dir == '' ? {} : {
    'CUDA_cuda_driver_LIBRARY': driver_library_dir / 'libcuda.so',
    'CUDA_nvml_LIBRARY': driver_library_dir / 'libnvidia-ml.so',
}
cccl_dep = subproject(
    'cccl',
    default_options: {'driver_library_dir': driver_library_dir},
).get_variable('cccl_dep')
libdeflate_dep = subproject('libdeflate').get_variable('libdeflate_dep')

cuda_compiler = meson.get_compiler('cuda')
# cudaMemcpyBatchAsync took its current signature (no failIdx) in CUDA 13.0.
if not cuda_compiler.version().version_compare('>= 13.0')
    error('cuDDL needs CUDA 13.0 or newer, found ' + cuda_compiler.version())
endif

# nvCOMP is optional; sketch construction still requires CUDA.
nvcomp_dep = cuda_compiler.find_library('nvcomp', required: get_option('nvcomp'))
nvcomp_enabled = nvcomp_dep.found() and cuda_compiler.compiles(
    '''#include <nvcomp/gzip.h>
int main() { return 0; }
''',
    name: 'nvCOMP headers',
)
if get_option('nvcomp').enabled() and not nvcomp_enabled
    error('nvCOMP was enabled but its headers are unavailable')
endif

cuddl_dep = declare_dependency(
    include_directories: include_directories('include'),
    dependencies: [cccl_dep, libdeflate_dep] + (nvcomp_enabled ? [nvcomp_dep] : []),
    compile_args: [
        '-DCUDDL_HAS_NVCOMP=' + (nvcomp_enabled ? '1' : '0'),
        '-DCUDDL_ABLATIONS=' + (get_option('ablations') ? '1' : '0'),
    ],
)

# MPI is optional; only include/cuddl/distributed.cuh needs it.
mpi_dep = dependency('mpi', language: 'cpp', required: false)
cuddl_distributed_dep = mpi_dep.found() ? declare_dependency(dependencies: [cuddl_dep, mpi_dep]) : disabler()

build_examples = (
    get_option('examples').enabled()
    or (get_option('examples').auto() and not meson.is_subproject())
)

build_benchmarks = (
    get_option('benchmarks').enabled()
    or (get_option('benchmarks').auto() and not meson.is_subproject())
)
build_tests = (
    get_option('tests').enabled()
    or (get_option('tests').auto() and not meson.is_subproject())
)

if build_benchmarks or build_examples
    cli11_dep = subproject('cli11').get_variable('cli11_dep')
endif

# BBTools is a pure-Java release vendored as a Meson subproject; its shell
# wrappers (comparesketch.sh, loglog.sh) serve as the BBTools metric oracle for
# benchmark acceptance. Only needed when benchmarks build.
bbtools_dir = ''
nvbench_dep = disabler()
if build_benchmarks
    bbtools_dir = subproject('bbtools').get_variable('bbtools_dir')

    # Header-only JSON used by the RefSeq parity command's machine-readable evidence output.
    nlohmann_json_dep = subproject('nlohmann_json').get_variable('nlohmann_json_dep')

    cmake = import('cmake')

    nvbench_opts = cmake.subproject_options()
    # NVCC cannot use the GNU dialect inferred from imported C++ target options.
    nvbench_opts.set_override_option('cuda_std', 'c++17')
    nvbench_opts.add_cmake_defines({
        'CMAKE_CXX_STANDARD': '20',
        'CCCL_DIR': meson.project_source_root() / 'subprojects/cccl/lib/cmake/cccl',
        'CMAKE_PROJECT_NVBench_INCLUDE': meson.project_source_root() / 'subprojects/packagefiles/nvbench/cccl.cmake',
        # nlohmann_json is already vendored as a Meson subproject; hand it to CPM
        # directly. Its origin hash bakes in the patch command's toolchain
        # versions, so a shared CPM_SOURCE_CACHE never hits across machines,
        # and the URL zip redirects to a host the site proxy blocks.
        'CPM_nlohmann_json_SOURCE': meson.project_source_root() / 'subprojects/nlohmann_json-3.12.0',
        # Same vendored source, exposed for the nvbench shim below.
        'NVBENCH_JSON_INCLUDE': meson.project_source_root() / 'subprojects/nlohmann_json-3.12.0/include',
    })
    # nvbench loads rapids-cmake through FetchContent before project(), so neither
    # CPM_SOURCE_CACHE nor toolchain variables reach it; only a cache define does.
    if get_option('rapids_cmake_source') != ''
        nvbench_opts.add_cmake_defines({
            'FETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE': get_option('rapids_cmake_source'),
        })
    endif
    nvbench_opts.add_cmake_defines(cuda_driver_cmake_defines)
    # nvbench is imported as a CMake subproject; its `nvbench` and `nvbench.main` targets are
    # exposed through Meson's cmake module. NVML and CUPTI are provided by the dev shell's
    # CUDA toolkit. C++20 is forced: building at nvbench's default C++17 trips a GCC 15.3
    # consteval-lambda regression in `state.cxx`.
    nvbench_cmake = cmake.subproject(
        'nvbench',
        required: true,
        options: nvbench_opts,
    )
    nvbench_dep = nvbench_cmake.dependency('nvbench')
    nvbench_main = nvbench_cmake.dependency('nvbench.main')
endif

if build_examples
    subdir('examples')
endif

if build_benchmarks
    subdir('benchmarks')
endif

if build_tests
    subdir('tests')
endif
