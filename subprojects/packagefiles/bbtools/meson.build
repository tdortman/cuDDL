project(
    'bbtools',
    version: '40.02',
    meson_version: '>= 1.3.0',
)

# The BBTools release is a pure-Java distribution: shell wrappers plus a prebuilt
# bbtools.jar and a resources tree. The wrap archive's single lead directory is
# `bbmap`, so this subproject's root *is* the extracted bbmap tree and each shell
# wrapper resolves its own directory at runtime (via $0), expecting bbtools.jar /
# resources as siblings. Consumers invoke the wrappers from this source directory
# rather than a flattened prefix. Java is supplied by the surrounding environment.

bbtools_dir = meson.current_source_dir()

bbtools_dep = declare_dependency()

meson.override_dependency(
    'bbtools',
    bbtools_dep,
)
