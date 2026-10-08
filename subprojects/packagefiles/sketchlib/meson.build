project(
    'sketchlib',
    version: '0.4.1',
    meson_version: '>= 1.3.0',
)

cargo = find_program('cargo')
cargo_target_dir = meson.current_build_dir() / 'cargo-target'

# Built for the host CPU, as the README recommends for performance.
sketchlib_binary = custom_target(
    'sketchlib',
    output: 'sketchlib',
    command: [
        'sh',
        '-c',
        'export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"; '
        + '"$1" build --release --locked --offline --bin sketchlib --manifest-path "$2" '
        + '--target-dir "$3" >"$4.log" 2>&1 || { cat "$4.log"; exit 1; }; '
        + 'cp "$3/release/sketchlib" "$4"',
        '_',
        cargo,
        meson.current_source_dir() / 'Cargo.toml',
        cargo_target_dir,
        '@OUTPUT@',
    ],
    console: true,
    build_by_default: true,
    build_always_stale: true,
    env: {'CARGO_NET_OFFLINE': 'true'},
)

meson.override_dependency('sketchlib', declare_dependency())
