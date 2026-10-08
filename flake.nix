{
  description = "DynamicDemiLog";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
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
            overlays = [ rust-overlay.overlays.default ];
          };

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" ];
          };

          lib = pkgs.lib;
          cudaPkgs = pkgs.cudaPackages_13_4;
          llvmPkgs = pkgs.llvmPackages_22;

          cudaToolkit = pkgs.symlinkJoin {
            name = "cuda-toolkit";

            paths = with cudaPkgs; [
              cuda_nvcc
              cuda_crt
              cuda_cudart
              cuda_profiler_api.include
              cuda_sanitizer_api
              cuda_cuobjdump
              cuda_nvdisasm

              cuda_gdb.bin
              nsight_systems
              nsight_compute

              # nvCOMP ships its headers, library and CMake config in separate
              # outputs, so the merge takes each of them: the default output is
              # the licence text and leaves the toolkit root empty of nvcomp.
              nvcomp.include
              nvcomp.lib
              nvcomp.dev

              # NVML and CUPTI are required by nvbench
              # (benchmark GPU monitoring).
              cuda_nvml_dev.include
              cuda_nvml_dev.stubs
              cuda_cupti.lib
              cuda_cupti.include

              # Required for clangd to correctly understand some of the
              # CUDA/STL headers when cuda_crt is present.
              libcurand.include

              # cuBLAS (lib + headers); cudarc links cublasLt.
              libcublas.lib
              libcurand.lib
              cuda_nvrtc.lib
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
            pkgs.zlib

            # OpenMP headers + libomp runtime.
            llvmPkgs.openmp
          ];

          nativeBuildInputs = with pkgs; [
            rustToolchain
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

            # BBTools is vendored as a Meson subproject (pure Java, no Nix
            # store path needed); only its JRE runtime belongs in the shell.
            jre_headless
          ];
        in
        {
          default = pkgs.mkShell {
            inherit buildInputs nativeBuildInputs;

            env = {
              CPATH = lib.makeIncludePath [
                cuda.path
                llvmPkgs.openmp
              ];

              CUDA_HOME = cuda.path;

              LD_LIBRARY_PATH = "${
                lib.makeLibraryPath (buildInputs ++ nativeBuildInputs)
              }:/run/opengl-driver/lib";
            };

            shellHook = ''
              # Local CPU benchmarks use RabbitSketch's native-ISA build.
              unset NIX_ENFORCE_NO_NATIVE

              export PATH="${cuda.path}/compute-sanitizer:$PATH"
              export PYTHONPATH="$(pwd)/scripts''${PYTHONPATH:+:$PYTHONPATH}"

              if [ ! -e .clangd ]; then
                cat > .clangd <<EOF
              CompileFlags:
                Compiler: ${llvmPkgs.clang}/bin/clang++
                Add:
                  - -std=c++20
                  - -fopenmp
                  - -D__INTELLISENSE__
                  - -D__CLANGD__
                  - -DCUDDL_HAS_NVCOMP=1
                  - -I$(pwd)/subprojects/cccl/libcudacxx/include
                  - -I$(pwd)/subprojects/cccl/cub
                  - -I$(pwd)/subprojects/cccl/thrust
                  - -I${cuda.path}/include
                  - -I${llvmPkgs.openmp}/include
                  - -I$(pwd)/include
                  - -I$(pwd)/subprojects/nvbench
                  - -I$(pwd)/subprojects/cuco/include
                  - -I$(pwd)/subprojects/libdeflate
                  - -I$(pwd)/subprojects/googletest-1.17.0/googletest/include

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

              ---

              If:
                PathMatch: .*\.(cu|cuh)$

              CompileFlags:
                Add:
                  - -xcuda
                  - --cuda-path=${cuda.path}
                  - --cuda-gpu-arch=${cuda.smTarget}
                  - -D__LIBCUDACXX__STD_VER=${cuda.version.major}
                  - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
                  - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
                  - -D__CUDA_ARCH__=${cuda.arch}
                  - -D__CUDACC_EXTENDED_LAMBDA__

              Diagnostics:
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
