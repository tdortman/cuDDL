project(
    'dashing2',
    version: '2.1.20',
    meson_version: '>= 1.3.0',
)

make = find_program('make')

# Upstream is Makefile-only. Sources plus all recursive submodules except
# fmt ship in this overlay; fmt is replaced by 11.2.0 because pinned
# fmt 8.0.1 fails consteval on current compilers.
dashing2_bin = custom_target(
    'dashing2',
    output: 'dashing2',
    command: [
        'sh',
        '-c',
        '"$1" -C "$2" dashing2 -j"$(nproc)" >"$3.log" 2>&1 || { cat "$3.log"; exit 1; }; cp "$2/dashing2" "$3"',
        '_',
        make,
        meson.current_source_dir(),
        '@OUTPUT@',
    ],
    console: true,
    build_by_default: true,
    build_always_stale: true,
)

meson.override_dependency('dashing2', declare_dependency())
