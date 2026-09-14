project(
    'libdeflate',
    'c',
    version: '1.26',
    meson_version: '>= 1.3.0',
)

# Upstream builds with CMake or its own Makefile. Consumers here need gzip
# decompression only, so this overlay compiles the decompressor, the gzip
# wrapper, the CRC32 used to verify members, and the CPU feature probes.
# Both feature probes build on every architecture; each guards its own body.
# Compression, zlib-wrapped streams, and the RISC-V matchfinder are omitted.
libdeflate_sources = files(
    'lib/utils.c',
    'lib/deflate_decompress.c',
    'lib/crc32.c',
    'lib/gzip_decompress.c',
    'lib/arm/cpu_features.c',
    'lib/x86/cpu_features.c',
)

libdeflate_lib = static_library(
    'libdeflate',
    libdeflate_sources,
    include_directories: include_directories('.'),
    install: false,
)

libdeflate_dep = declare_dependency(
    link_with: libdeflate_lib,
    include_directories: include_directories('.'),
)

meson.override_dependency('libdeflate', libdeflate_dep)
