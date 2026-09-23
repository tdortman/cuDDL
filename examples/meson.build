# One unit per bucket count and range of eight k-mer lengths bounds per-unit compiler memory.
index_dispatch_sources = []
foreach buckets : [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    foreach first_k : [1, 9, 17, 25]
        instantiations = []
        foreach k : range(first_k, first_k == 25 ? 32 : first_k + 8)
            instantiations += 'template void dispatch_reference_index<@0@, @1@>(reference_index_command const&);'.format(
                buckets,
                k,
            )
        endforeach
        index_dispatch_sources += configure_file(
            input: 'reference_index_dispatch.cu.in',
            output: 'reference_index_@0@_k@1@.cu'.format(buckets, first_k),
            configuration: {'INSTANTIATIONS': '\n'.join(instantiations)},
        )
    endforeach
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
