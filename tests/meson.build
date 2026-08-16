gtest_dep = subproject('gtest').get_variable('gtest_dep')
gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')

test_exe = executable(
    'test-cuddl',
    files('cuddl_test.cu'),
    dependencies: [cuddl_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

test('cuddl-tests', test_exe)
