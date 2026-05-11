# Contributing to NeurIPS 2025 GPU Kernels Tutorial

Thanks for your interest in this tutorial! This guide explains how to contribute, report issues, and submit changes.

## Before You Start

- Read `README.md` to understand the tutorial scope (HIP, Triton, and GEAK agent-based kernel optimization).
- Skim the tutorial PDFs at the repo root for context on the lesson flow.
- Ensure you have a supported GPU environment (AMD GPU with ROCm 7.0+ recommended) to run the kernels.

## Development Setup

This repo is designed to run inside a Docker container with all dependencies pre-installed.

```bash
# Initialize git submodules (GEAK-agent / GEAK-eval are pulled in via Dockerfile)
git submodule update --init --recursive

# Build the tutorial container
docker build -t neurips-gpu-tutorial .

# Run with GPU access and Jupyter exposed
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --network=host \
  -v "$(pwd)":/app/jupyter \
  neurips-gpu-tutorial
```

For HIP-only examples you can also build the C++ files directly with `hipcc` outside Docker.

## Workflow

1. Create a new branch from `main`.
2. Keep changes focused and scoped — prefer one concept per notebook/section.
3. Verify notebooks run end-to-end in a freshly built container before submitting.
4. Open a Pull Request with motivation, impact, and verification steps.

## Code Style and Quality

- Follow PEP 8 for Python code; use clear cell-level comments in notebooks.
- For HIP/C++ examples, keep naive and optimized versions side-by-side and label them clearly.
- Clear all cell outputs in notebooks before committing (unless an output is pedagogically important).
- Add documentation or markdown cells when intent is non-obvious.

## Testing and Verification

This project depends on GPU hardware/drivers. In your PR, include:

- Test environment (GPU model, ROCm version, container or host run mode)
- Notebook(s) executed end-to-end
- For HIP examples: `hipcc` compile output and a sample run

```bash
# Example: compile and run an HIP example
cd src/hip/01-memory-coalescing
hipcc -O3 example.cpp -o example && ./example
```

## Filing Issues

Please include:

- Reproduction steps (Docker build/run command, notebook path, cell range)
- Expected vs actual behavior
- Environment (OS, GPU, ROCm version, Docker version, Python version)
- Relevant logs or a minimal repro

## Security

If you discover a security issue, do not open a public issue. Contact maintainers through a private channel.

The Dockerfile clones external repos (`GEAK-agent`, `GEAK-eval`) at build time — flag any supply-chain or credential concerns privately.

## Suggested Contributions

- Add new HIP optimization examples (e.g., shared memory, tensor cores)
- Extend Triton tutorials with new fused operators
- Improve GEAK notebook usability or add new agent recipes
- Improve docs, slides, and reproducibility instructions

## License

By contributing, you agree that your contributions are licensed under the repository `LICENSE` (MIT).
