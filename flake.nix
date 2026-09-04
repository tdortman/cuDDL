{
  description = "DynamicDemiLog";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    {
      devShells = nixpkgs.lib.genAttrs supportedSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          lib = pkgs.lib;
          cudaPkgs = pkgs.cudaPackages_13_3;
          llvmPkgs = pkgs.llvmPackages_22;

          # Compute Sanitizer 2026.3 adds initcheck support for CCCL's batched copies.
          cudaSanitizer = cudaPkgs.cuda_sanitizer_api.overrideAttrs (
            final: _: {
              version = "13.4.46";
              src =
                let
                  platform = if system == "x86_64-linux" then "linux-x86_64" else "linux-sbsa";
                in
                pkgs.fetchurl {
                  url = "https://packages.nvidia.com/bin-archive/pool/${platform}/5B515474-7E78-11F1-8656-C51E4F4B317F/cuda_sanitizer_api-${platform}-${final.version}-archive.tar.xz";
                  sha256 =
                    {
                      x86_64-linux = "cbffa4277abe42dd451519b1e367176415d71925e061356780291ccb58116869";
                      aarch64-linux = "880b5165f88318599c037730b62ceca7db2a0a7b24dc34ebed00acd784459209";
                    }
                    .${system};
                };
            }
          );

          cudaToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit";
            paths = with cudaPkgs; [
              cuda_nvcc
              cuda_crt
              cuda_cudart
              cuda_profiler_api.include
              cuda_cuobjdump
              cuda_nvdisasm

              cuda_gdb.bin
              nsight_systems
              nsight_compute
              cudaSanitizer

              # NVML and CUPTI are required by nvbench (benchmark GPU monitoring).
              cuda_nvml_dev.include # nvml.h
              cuda_nvml_dev.stubs # libnvidia-ml.so stub
              cuda_cupti.lib # libcupti.so
              cuda_cupti.include # cupti.h

              # I do not know why cuRAND headers are necessary
              # for clangd to not freak out about STL headers when cuda_crt is
              # also present but at least it's a somewhat cheap dependency...
              libcurand.include
            ];
          };

          cuda = {
            arch = "1200";
            smTarget = "sm_120";
            path = cudaToolkit;
            version = {
              complete = cudaPkgs.cudaMajorMinorVersion;
              major = cudaPkgs.cudaMajorVersion;
              minor = lib.lists.last (builtins.splitVersion cuda.version.complete);
            };
          };

          buildInputs = [
            cudaToolkit
            pkgs.stdenv.cc.cc.lib
          ];

          nativeBuildInputs = with pkgs; [
            llvmPkgs.clang-tools
            llvmPkgs.clang
            meson
            uv
            pkg-config
            doxygen
            graphviz

            ninja
            cmake

            texliveFull
            tex-fmt

            # BBTools is vendored as a Meson subproject (pure Java, no Nix store
            # path needed); only its JRE runtime belongs in the dev shell.
            jre_headless
          ];
        in
        {
          default = pkgs.mkShell {
            inherit buildInputs nativeBuildInputs;

            env = {
              CPATH = lib.makeIncludePath [ cuda.path ];
              CUDA_HOME = cuda.path;

              LD_LIBRARY_PATH = "${
                lib.makeLibraryPath (buildInputs ++ nativeBuildInputs)
              }:/run/opengl-driver/lib";
            };

            shellHook = ''
                  export PATH="${cuda.path}/compute-sanitizer:$PATH"
                  export PYTHONPATH=$(pwd)/scripts:$PYTHONPATH
                  if [ ! -e .clangd ]; then
                    cat > .clangd <<EOF
              CompileFlags:
                Compiler: ${cuda.path}/bin/nvcc
                Add:
                  - -std=c++20
                  - -xcuda
                  - --cuda-path=${cuda.path}
                  - -D__INTELLISENSE__
                  - -D__CLANGD__
                  - -I$(pwd)/subprojects/cccl/libcudacxx/include
                  - -I$(pwd)/subprojects/cccl/cub
                  - -I$(pwd)/subprojects/cccl/thrust
                  - -I${cuda.path}/include
                  - -I$(pwd)/include
                  - -I$(pwd)/subprojects/nvbench
                  - -I$(pwd)/subprojects/cuco/include
                  - -I$(pwd)/subprojects/googletest-1.17.0/googletest/include
                  - -D__LIBCUDAXX__STD_VER=${cuda.version.major}
                  - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
                  - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
                  - -D__CUDA_ARCH__=${cuda.arch}
                  - --cuda-gpu-arch=${cuda.smTarget}
                  - -D__CUDACC_EXTENDED_LAMBDA__
                  - -DPARAM_SWEEP_GROUP
                Remove:
                  - -Xcompiler=*
                  - -G
                  - "-arch=*"
                  - "-Xfatbin*"
                  - "-Xnvlink*"
                  - "-gencode*"
                  - "--generate-code*"
                  - "--generate-line-info"
                  - "--compiler-options*"
                  - "--expt-extended-lambda"
                  - "--expt-relaxed-constexpr"
                  - "-forward-unknown-to-host-compiler"
                  - "-Werror=cross-execution-space-call"

              Diagnostics:
                UnusedIncludes: None
                Suppress:
                  - variadic_device_fn
                  - attributes_not_allowed
                  - undeclared_var_use_suggest
                  - typename_invalid_functionspec
                  - expected_expression
                  - deduction_guide_target_attr
              EOF
                    echo ".clangd created by flake shellHook"
                  fi
            '';
          };
        }
      );
    };
}
