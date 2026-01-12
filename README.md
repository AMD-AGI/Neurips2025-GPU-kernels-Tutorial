# NeurIPS 2025 GPU Kernels Tutorial

Tutorial materials and hands-on exercises for GPU kernel optimization on GPUs using HIP, Triton, and AI-assisted optimization techniques.

## Overview

This repository contains comprehensive materials for learning GPU kernel optimization, including:
- Low-level HIP/C++ implementations demonstrating optimization techniques
- High-level Triton kernel development tutorials
- AI-powered kernel optimization using GEAK (GPU Kernel Optimization Agent)

## Contents

### HIP Examples (`src/hip/`)
C++ kernel implementations with naive and optimized versions:
- **01-memory-coalescing**: Optimizing memory access patterns
- **02-loop-unrolling**: A comparison case using unrolling. 
- **Hands_On_Kernels_and_Optmiztion.ipynb**: Interactive tutorial notebook

### Triton Examples (`src/triton/`)
Python-based kernel optimization tutorials:
- Fused softmax implementation
- Layer normalization kernels
- Comprehensive Triton optimization guide

### GEAK (`src/geak/`)
Agent-based kernel optimization framework for automated kernel tuning and optimization.

### Tutorial Materials
- `Neurips_tutorial.pdf`: Complete tutorial documentation
- `Neurips_tutorial.pptx`: Presentation slides

## Quick Start

1. **HIP Examples**: Navigate to `src/hip/` and compile the C++ files using ROCm toolchain
2. **Triton Examples**: Open the Jupyter notebooks in `src/triton/` (requires Triton installation)
3. **GEAK**: Start with `src/geak/Main.ipynb` for agent-based optimization

## Requirements

- ROCm toolkit (for HIP examples)
- Python with Jupyter (for Triton and GEAK examples)
- AMD GPU with ROCm support

## License

MIT License - see [LICENSE](LICENSE) for details.

