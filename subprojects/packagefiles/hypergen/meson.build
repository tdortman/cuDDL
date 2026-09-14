project(
    'hypergen',
    version: '0.2.2',
    meson_version: '>= 1.3.0',
)

cargo = find_program('cargo')
cargo_target_dir = meson.current_build_dir() / 'cargo-target'

# GPU arch is one cargo feature; override for non-Blackwell hosts, empty for CPU-only.
hypergen_feature = run_command(
    'sh',
    '-c',
    'echo "${HYPERGEN_CUDA_FEATURE:-cuda-sketch-blackwell}"',
    check: true,
).stdout().strip()

hypergen_bin = custom_target(
    'hyper-gen',
    output: 'hyper-gen',
    command: [
        'sh',
        '-c',
        'export CUDA_ROOT="${CUDA_ROOT:-$CUDA_HOME}"; '
        + 'export RUSTFLAGS="${RUSTFLAGS:-} -L native=$CUDA_ROOT/lib/stubs"; '
        + 'feature="$1"; cargo_bin="$2"; manifest="$3"; targetdir="$4"; out="$5"; '
        + 'if [ -n "$feature" ]; then set -- --features "$feature"; else set --; fi; '
        + '"$cargo_bin" build --release --locked --offline --bin hyper-gen "$@" --manifest-path "$manifest" '
        + '--target-dir "$targetdir" >"$out.log" 2>&1 '
        + '|| { cat "$out.log"; exit 1; }; cp "$targetdir/release/hyper-gen" "$out"',
        '_',
        hypergen_feature,
        cargo,
        meson.current_source_dir() / 'Cargo.toml',
        cargo_target_dir,
        '@OUTPUT@',
    ],
    console: true,
    build_by_default: true,
    build_always_stale: true,
)

meson.override_dependency('hypergen', declare_dependency())
