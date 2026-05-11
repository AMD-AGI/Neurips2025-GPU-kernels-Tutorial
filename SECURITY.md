# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **GitHub Private Vulnerability Reporting:** [Report a vulnerability](https://github.com/AMD-AGI/Neurips2025-GPU-kernels-Tutorial/security/advisories/new)
- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository — tutorial materials (HIP, Triton, GEAK notebooks), the `Dockerfile`, and the example kernels under `src/`.

Note: the `Dockerfile` clones external AMD-AGI repositories (`GEAK-agent`, `GEAK-eval`) at build time. Issues with those upstream repos should be reported in their respective security policies. Please flag any supply-chain, Docker build, or notebook-execution concerns affecting this tutorial repo privately.

For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).
