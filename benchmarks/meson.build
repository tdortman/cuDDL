# Raw-FASTA end-to-end gate binary. Per the benchmark contract this is a dedicated CLI11
# target (it measures the full parse -> sketch -> compare pipeline), not a google-benchmark one.
executable(
    'cuddl-benchmark-fasta',
    'cuddl_benchmark_fasta.cu',
    dependencies: [cuddl_dep, cli11_dep],
    cuda_args: cuda_args,
    install: false,
)
