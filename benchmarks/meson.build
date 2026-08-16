benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')
cuco_dep = subproject('cuco').get_variable('cuco_dep')

bench_deps = [benchmark_dep, cuddl_dep]

benchmarks = {}

foreach name, config : benchmarks
    extra_cuda_args = config.get('extra_args', [])
    deps = config.get('deps', [])
    executable(
        name,
        config['src'],
        dependencies: bench_deps + deps,
        cuda_args: cuda_args + extra_cuda_args,
        install: false,
    )
endforeach