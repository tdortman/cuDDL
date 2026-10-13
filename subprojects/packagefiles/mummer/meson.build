project(
    'mummer',
    version: '4.0.1',
    meson_version: '>= 1.3.0',
)

make = find_program('make')

# Upstream is autotools. The release tarball ships its configure script; static linking keeps
# libtool from turning the binaries into wrapper scripts that only run from the build tree.
mummer_bins = custom_target(
    'mummer',
    output: ['nucmer', 'delta-filter'],
    command: [
        'sh',
        '-c',
        'src="$1"; make="$2"; build="$3"; shift 3; mkdir -p "$build" && ( cd "$build" '
        + '&& { [ -f Makefile ] || "$src/configure" --disable-shared CXXFLAGS="-O3 -march=native" '
        + 'CFLAGS="-O3 -march=native"; } >configure.log 2>&1 '
        + '&& "$make" -j"$(nproc)" nucmer delta-filter >make.log 2>&1 '
        + '|| { cat configure.log make.log 2>/dev/null; exit 1; } ) || exit 1; '
        + 'cp "$build/nucmer" "$1" && cp "$build/delta-filter" "$2"',
        '_',
        meson.current_source_dir(),
        make,
        meson.current_build_dir() / 'autotools',
        '@OUTPUT0@',
        '@OUTPUT1@',
    ],
    console: true,
    build_by_default: true,
    build_always_stale: true,
)

meson.override_dependency('mummer', declare_dependency())
