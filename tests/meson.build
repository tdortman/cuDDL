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

index_file_test_exe = executable(
    'test-reference-index-file',
    files('reference_index_file_test.cu'),
    dependencies: [cuddl_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)
test('reference-index-file-tests', index_file_test_exe)

blacklist_test_exe = executable(
    'test-blacklist',
    files('blacklist_test.cu'),
    dependencies: [cuddl_dep, gtest_dep, gtest_main_dep],
    cuda_args: cuda_args,
    install: false,
)
test('blacklist-tests', blacklist_test_exe)

result_json_test_exe = executable(
    'test-result-json',
    files('result_json_test.cu'),
    dependencies: [cccl_dep, gtest_dep, gtest_main_dep, nlohmann_json_test_dep],
    include_directories: include_directories('../benchmarks'),
    cuda_args: cuda_args,
    install: false,
)

test('result-json-tests', result_json_test_exe)

resident_sequence_test_exe = executable(
    'test-resident-sequence',
    files('resident_sequence_test.cu'),
    dependencies: [cuddl_dep, gtest_dep, gtest_main_dep],
    include_directories: include_directories('../benchmarks'),
    cuda_args: cuda_args,
    install: false,
)
test('resident-sequence-tests', resident_sequence_test_exe)

# Distributed sketching across MPI ranks; the ranks may share one GPU.
mpiexec = find_program('mpiexec', required: false)
if mpi_dep.found() and mpiexec.found()
    distributed_test_exe = executable(
        'test-distributed',
        files('distributed_test.cu'),
        dependencies: [cuddl_distributed_dep, gtest_dep],
        cuda_args: cuda_args,
        install: false,
    )
    foreach ranks : [1, 2, 3]
        test(
            'distributed-tests-@0@-ranks'.format(ranks),
            mpiexec,
            args: ['-n', ranks.to_string(), distributed_test_exe],
            is_parallel: false,
            timeout: 300,
        )
    endforeach
endif
