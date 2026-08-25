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

# Raw-FASTA end-to-end gate binary. Per the benchmark contract this is a dedicated CLI11
# target (it measures the full parse -> sketch -> compare pipeline), not a google-benchmark one.
executable(
    'cuddl-benchmark-fasta',
    'cuddl_benchmark_fasta.cu',
    dependencies: [cuddl_dep, cli11_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cuddl-dynamic-demilog-benchmark',
    'cuddl_dynamic_demilog_benchmark.cu',
    dependencies: [cuddl_dep, cli11_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cuddl-compact-search-benchmark',
    'cuddl_compact_search_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main],
    cuda_args: cuda_args,
    install: false,
)

# Provides its own main (strips --summary-csv, runs nvbench, then writes the summary CSV).
# nvbench_main stays linked: its static archive member is skipped when main is already
# defined, while its cuda driver/nvml/cupti link dependencies are still required.
executable(
    'cuddl-search-decision-benchmark',
    'cuddl_search_decision_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main],
    cuda_args: cuda_args,
    install: false,
)

# BBTools RefSeq DDL parity validation. A CLI11 contract target, not an nvbench one: it verifies
# the official asset's provenance, decodes the A48 rows, runs the independent host exhaustive
# oracle, builds and searches the cuDDL GPU index, drives the first-party BBTools DDLIndex/CSR2
# Java harness, and emits machine-readable JSON evidence comparing every surface (or reports the
# exact external prerequisite when the official RefSeq asset is not available).
executable(
    'cuddl-refseq-parity',
    'cuddl_refseq_parity.cu',
    dependencies: [cuddl_dep, cli11_dep, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)
