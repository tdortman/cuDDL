gtest_dep = subproject('gtest').get_variable('gtest_dep')
gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')
benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')

test_exe = executable(
    'test-cuddl',
    [],
    dependencies: [cuddl_dep, benchmark_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

test('cuddl-tests', test_exe)

if build_benchmarks
    python = import('python').find_installation()
    test(
        'construction-bakeoff-equivalence',
        python,
        args: [
            files('construction_bakeoff_test.py'),
            meson.project_build_root() / 'benchmarks' / 'construction-bakeoff',
        ],
    )
    test(
        'comparison-bakeoff-equivalence',
        python,
        args: [
            files('comparison_bakeoff_test.py'),
            meson.project_build_root() / 'benchmarks' / 'comparison-bakeoff',
        ],
    )
endif