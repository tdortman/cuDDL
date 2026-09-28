# The CLIs support one sketch configuration each build; reconfigure with -Dcli_kmer_length,
# -Dcli_buckets and -Dcli_exponent_bits to read databases built with other settings.
cli_args = cuda_args + [
    '-DCUDDL_CLI_KMER_LENGTH=@0@'.format(get_option('cli_kmer_length')),
    '-DCUDDL_CLI_BUCKETS=@0@'.format(get_option('cli_buckets')),
    '-DCUDDL_CLI_EXPONENT_BITS=@0@'.format(get_option('cli_exponent_bits')),
]

examples = {
    'cuddl-reference-index': {'src': 'reference_index.cu', 'args': cli_args},
    'cuddl-build-reference-db': {'src': 'build_reference_database.cu', 'args': cli_args},
    'cuddl-main': {'src': 'main.cu', 'args': cuda_args},
}

foreach name, config : examples
    executable(
        name,
        config['src'],
        dependencies: [cuddl_dep, cli11_dep],
        cuda_args: config['args'],
        install: true,
    )
endforeach
