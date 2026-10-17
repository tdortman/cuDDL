project(
    'pp-sketchlib',
    'c',
    'cpp',
    version: '2.2.0',
    meson_version: '>= 1.3.0',
)

# The library's C++ and CUDA sources without its Python bindings and web build, for a harness to
# compile with its own CUDA architectures. The no-python patch replaces the few Python calls the
# library makes for interrupts and warnings.
pp_sketchlib_sources = files(
    'src/api.cpp',
    'src/database/database.cpp',
    'src/dist/dist.cpp',
    'src/dist/linear_regression.cpp',
    'src/dist/matrix_ops.cpp',
    'src/gpu/device_memory.cu',
    'src/gpu/device_reads.cu',
    'src/gpu/dist.cu',
    'src/gpu/gpu_api.cpp',
    'src/gpu/gpu_countmin.cu',
    'src/gpu/sketch.cu',
    'src/random/kmeans/KMeansRexCore.cpp',
    'src/random/kmeans/mersenneTwister2002.c',
    'src/random/random_match.cpp',
    'src/random/rng.cpp',
    'src/reference.cpp',
    'src/sketch/countmin.cpp',
    'src/sketch/seqio.cpp',
    'src/sketch/sketch.cpp',
)

# Databases record a hash of the sketching code; upstream's Makefile hashes the same files.
sketch_hash = run_command(
    'sh',
    '-c',
    'cat src/sketch/*.cpp src/sketch/*.hpp src/gpu/sketch.cu | sha1sum | cut -d" " -f1',
    check: true,
).stdout().strip()
configure_file(
    output: 'version.h',
    configuration: {'SKETCH_VERSION': '"' + sketch_hash + '"'},
    output_format: 'c',
)

pp_sketchlib_include = include_directories('.', 'src', 'vendor/highfive/include')
pp_sketchlib_args = ['-DGPU_AVAILABLE', '-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS']
