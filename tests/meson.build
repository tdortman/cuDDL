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