examples = {
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
