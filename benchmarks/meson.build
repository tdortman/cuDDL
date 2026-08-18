# Raw-FASTA end-to-end gate binary. Per the benchmark contract this is a dedicated CLI11
# target (it measures the full parse -> sketch -> compare pipeline), not a google-benchmark one.
executable(
    'cuddl-benchmark-fasta',
    'cuddl_benchmark_fasta.cu',
    dependencies: [cuddl_dep, cli11_dep],
    cuda_args: cuda_args,
    install: false,
)

# nvbench comparison of cuDDL construction and cardinality estimation against cuCollections
# HyperLogLog. Links nvbench::main so it brings its own entry point.
if build_benchmarks
    cuco_bench_dep = subproject('cuco').get_variable('cuco_dep')
    executable(
        'cuddl-hll-comparison',
        'cuddl_vs_cuco_hll.cu',
        dependencies: [cuddl_dep, cuco_bench_dep, nvbench_dep, nvbench_main],
        cuda_args: cuda_args,
        install: false,
    )

    executable(
        'cuddl-hll-accuracy',
        'cuddl_hll_accuracy.cu',
        dependencies: [cuddl_dep, cuco_bench_dep, cli11_dep],
        cuda_args: cuda_args,
        install: false,
    )
endif
