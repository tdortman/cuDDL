cmake = import('cmake')
rabbitsketch_opts = cmake.subproject_options()
rabbitsketch_opts.add_cmake_defines(
    {
        'CXXAPI': true,
        'RABBITSKETCH_BUILD_TESTS': false,
        'RABBITSKETCH_NATIVE_ARCH': true,
    },
)
rabbitsketch_project = cmake.subproject('rabbitsketch', options: rabbitsketch_opts)
rabbitsketch_dep = rabbitsketch_project.dependency('rabbitsketch_static')
rabbitsketch_revision = import('fs').read('../subprojects/rabbitsketch.wrap').split('revision = ')[1].split('\n')[0]
executable(
    'rabbitsketch-pairwise-accuracy',
    'rabbitsketch_pairwise_accuracy.cpp',
    cpp_args: ['-DRABBITSKETCH_REVISION="' + rabbitsketch_revision + '"'],
    dependencies: [rabbitsketch_dep, cli11_dep, nlohmann_json_dep],
    install: false,
)
executable(
    'rabbitsketch-pipeline-benchmark',
    'rabbitsketch_pipeline_benchmark.cpp',
    cpp_args: ['-DRABBITSKETCH_REVISION="' + rabbitsketch_revision + '"'],
    dependencies: [rabbitsketch_dep, cli11_dep, nlohmann_json_dep, cuddl_dep, nvbench_dep, dependency('openmp')],
    install: false,
)

cuco_bench_dep = subproject('cuco').get_variable('cuco_dep')

executable(
    'cuddl-reference-build-benchmark',
    'cuddl_reference_build_benchmark.cu',
    dependencies: [cuddl_dep, cli11_dep, nvbench_dep, nvbench_main, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

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
    dependencies: [cuddl_dep, cuco_bench_dep, cli11_dep, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cuddl-pairwise-accuracy',
    'cuddl_pairwise_accuracy.cu',
    dependencies: [cuddl_dep, cli11_dep, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cuddl-pairwise-batch-benchmark',
    'cuddl_pairwise_batch_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

# Raw-FASTA end-to-end gate binary. Per the benchmark contract this is a dedicated CLI11
# target (it measures the full parse -> sketch -> compare pipeline), not a google-benchmark one.
executable(
    'cuddl-benchmark-fasta',
    'cuddl_benchmark_fasta.cu',
    dependencies: [cuddl_dep, cli11_dep, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cuddl-dynamic-demilog-benchmark',
    'cuddl_dynamic_demilog_benchmark.cu',
    dependencies: [cuddl_dep, cli11_dep, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

# FASTA application and resident packed/sequence pipelines; GPU stages use NVBench.
executable(
    'cuddl-pipeline-benchmark',
    'cuddl_pipeline_benchmark.cu',
    dependencies: [cuddl_dep, cli11_dep, nvbench_dep, nvbench_main, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

executable(
    'cuddl-compact-search-benchmark',
    'cuddl_compact_search_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main, nlohmann_json_dep],
    cuda_args: cuda_args,
    install: false,
)

# Measures launch overhead and bounded-output indexed selection on resident inputs.
executable(
    'cuddl-search-overhead-benchmark',
    'cuddl_search_overhead_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main],
    cuda_args: cuda_args,
    install: false,
)

# Provides its own main (strips --summary-json, runs nvbench, then writes shared JSON).
# nvbench_main stays linked: its static archive member is skipped when main is already
# defined, while its cuda driver/nvml/cupti link dependencies are still required.
executable(
    'cuddl-search-decision-benchmark',
    'cuddl_search_decision_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main, nlohmann_json_dep],
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

# A/B experiments against the production kernels; each state validates before timing.
executable(
    'cuddl-efficiency-benchmark',
    'cuddl_efficiency_benchmark.cu',
    dependencies: [cuddl_dep, nvbench_dep, nvbench_main],
    cuda_args: cuda_args,
    install: false,
)
