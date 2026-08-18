project(
    'cuDDL',
    ['cuda', 'cpp'],
    version: '0.10',
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

# The FASTA parser reuses cuSBF's FASTX reader, pulled in as a pinned Meson subproject. cuSBF's
# exported `cusbf_dep` carries its headers and propagates its own zlib dependency, so no direct
# sibling include path or zlib wiring is needed here.
cuddl_dep = declare_dependency(
    include_directories: include_directories('include'),
    dependencies: [subproject('cusbf').get_variable('cusbf_dep')],
)

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
    cmake = import('cmake')

    nvbench_opts = cmake.subproject_options()
    nvbench_opts.add_cmake_defines({'CMAKE_CXX_STANDARD': '20'})
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
