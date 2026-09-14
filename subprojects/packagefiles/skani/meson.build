project(
    'skani',
    version: '0.3.2',
    meson_version: '>= 1.3.0',
)

cargo = find_program('cargo')
cargo_target_dir = meson.current_build_dir() / 'cargo-target'

skani_binary = custom_target(
    'skani',
    output: 'skani',
    command: [
        'sh',
        '-c',
        '"$1" build --release --locked --offline --bin skani --manifest-path "$2" --target-dir "$3" >"$4.log" 2>&1 || { cat "$4.log"; exit 1; }; cp "$3/release/skani" "$4"',
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

skani_dep = declare_dependency()
meson.override_dependency('skani', skani_dep)
