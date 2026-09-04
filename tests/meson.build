gtest_dep = subproject('gtest').get_variable('gtest_dep')
gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')
nlohmann_json_test_dep = subproject('nlohmann_json').get_variable('nlohmann_json_dep')

test_exe = executable(
    'test-cuddl',
    files('cuddl_test.cu'),
    dependencies: [cuddl_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)

test('cuddl-tests', test_exe)

result_json_test_exe = executable(
    'test-result-json',
    files('result_json_test.cu'),
    dependencies: [cccl_dep, gtest_dep, gtest_main_dep, nlohmann_json_test_dep],
    include_directories: include_directories('../benchmarks'),
    cuda_args: cuda_args,
    install: false,
)

test('result-json-tests', result_json_test_exe)
