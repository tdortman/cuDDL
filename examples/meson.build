index_dispatch_sources = []
foreach buckets : [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    index_dispatch_sources += configure_file(
        input: 'reference_index_dispatch.cu.in',
        output: 'reference_index_' + buckets.to_string() + '.cu',
        configuration: {'BUCKETS': buckets},
    )
endforeach

examples = {
    'cuddl-reference-index': {
        'src': ['reference_index.cu'] + index_dispatch_sources,
    },
    'cuddl-build-reference-db': {
        'src': 'build_reference_database.cu',
    },
    'cuddl-main': {
        'src': 'main.cu',
    },
}

foreach name, config : examples
    executable(
        name,
        config['src'],
        dependencies: [cuddl_dep, cli11_dep] + config.get('deps', []),
        cuda_args: cuda_args,
        install: true,
    )
endforeach
