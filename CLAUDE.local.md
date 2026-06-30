# GPU Kernel From Scratch — Project Memory
 
> Personal, gitignored project context for Claude Code. Add `CLAUDE.local.md` to
> `.gitignore`. Keep this file under ~200 lines and high-signal. Update the
> **Current Phase** and **What I've Tried** sections at the end of each work week.
 
## Goal
 
Implement attention from scratch in CUDA C++, culminating in a working
FlashAttention kernel, and benchmark it against PyTorch. Produce a public repo
plus a writeup documenting the optimization journey. Timeline: 10 weeks,
~12–15 hours/week.
 
This is a **learning project**. The point is that I understand every line, can
explain it in an interview, and come out of it competent in CUDA C++.
 
## How I want you to help me (read this first)
 
I am proficient in Python and Java but new to C++ and CUDA. I am doing this
partly to learn C++ for professional reasons. So:
 
- **Default to explaining, not generating.** When I'm stuck, explain the concept
  and let me write the code myself.
- When I ask for a kernel, first ask whether I want hints to attempt it myself,
  or a worked example I'll then re-implement from scratch.
- Explain compiler errors in terms of the underlying concept (pointers, memory
  layout, indexing) — not just the one-line fix.
- Review code I write, point out bugs, and explain *why* each is a bug.
- **Never** hand me a complete kernel to paste without understanding it. That
  defeats the entire purpose of the project.
- When suggesting an optimization, name the profiler metric that motivates it.
## Environment
 
- **Weeks 1–8 (learning, correct kernels):** free cloud GPU — Google Colab
  (NVIDIA T4, 16GB) or Kaggle (P100). No local NVIDIA GPU.
- **Weeks 9–10 (profiling, optimization):** rented hourly GPU (vast.ai / RunPod,
  RTX 3090 or 4090) for clean Nsight Compute access.
- **Compile arch depends on the card — update the build command when I switch:**
  T4 = `sm_75`, RTX 3090 = `sm_86`, RTX 4090 = `sm_89`.
- **Build:** `nvcc -O3 -arch=sm_75 src/kernels/<file>.cu -o bin/<name>`
- **Reference / validation:** PyTorch (compare every kernel's output to a
  PyTorch reference within float tolerance).
- **Profiling:** Nsight Compute (`ncu`) for per-kernel metrics; Nsight Systems
  for timelines.
- Files don't persist on cloud GPUs — commit and push to GitHub frequently.
## Conventions
 
- All kernels written from scratch. No cuBLAS / cuDNN wrappers for the core math.
- One file per kernel approach in `src/kernels/`.
- Every kernel has a matching CPU/PyTorch reference in `src/reference/`.
- Benchmark results saved as CSV in `benchmarks/results/`.
- Each week's decisions and learnings logged in `docs/decisions/week-N.md`.
- Starting numeric convention: fp16 storage, fp32 accumulation (revisit later).
## Repo layout
 
```
flashattention-in-cuda/
├── CLAUDE.local.md        # this file (gitignored)
├── src/kernels/           # CUDA kernels, one file per approach
├── src/reference/         # CPU / PyTorch reference implementations
├── benchmarks/            # benchmark scripts + results/
├── docs/decisions/        # week-N.md decision + learning logs
└── README.md              # project overview + final writeup
```
 
## Roadmap and milestones
 
Each week: a focus and a concrete "done" milestone. Treat *correct but slow* as a
real milestone before chasing speed.
 
- **Week 1 — C++ fundamentals + toolchain + vector add.**
  Pointers, malloc/free, headers, compile-link cycle. Toolchain working on Colab.
  *Done:* vector-add kernel matches a NumPy reference.
- **Week 2 — CUDA execution model + naive matmul.**
  threads → warps → blocks → grid; choosing block/grid dims.
  *Done:* naive matmul correct vs PyTorch, baseline runtime measured.
- **Week 3 — Shared memory + tiled matmul + first profiling.**
  Tiling for data reuse; install + read Nsight Compute.
  *Done:* tiled matmul shows clear multiple-x speedup, explained via a profiler
  memory-throughput metric.
- **Week 4 — Parallel reduction.**
  Tree reduction; warp divergence; shared-memory bank conflicts.
  *Done:* correct sum/max reduction with one measured optimization pass.
- **Week 5 — Softmax + numerical stability.**
  Reduction + elementwise; subtract-the-max trick.
  *Done:* stable softmax matching torch.softmax to float tolerance.
- **Week 6 — Naive attention.**
  softmax(QKᵀ/√d)V with the full N×N matrix materialized in global memory.
  *Done:* correct attention vs PyTorch; measured how much memory N×N eats.
- **Week 7 — FlashAttention, correct-but-slow.**
  Online softmax: running max + running sum, rescale by exp(old_max − new_max).
  *Done:* FlashAttention output matches the naive version numerically.
- **Week 8 — Debugging + hardening (buffer week).**
  Edge cases, varied sequence lengths, fp16/fp32 accumulation.
  *Done:* robustly correct across several input sizes; passes the test suite.
- **Week 9 — Profiling-driven optimization (rented GPU).**
  profile → hypothesize → change one thing → re-profile. Coalescing,
  bank-conflict padding, occupancy.
  *Done:* documented speedup over week 8 with before/after profiler evidence.
- **Week 10 — Final benchmarking + writeup.**
  Benchmark vs torch scaled_dot_product_attention across sequence lengths.
  *Done:* finished repo + writeup with the optimization journey and charts.
## Current Phase
 
**Week 1 — C++ fundamentals, toolchain setup, vector-add kernel.**
Running on: Colab free T4 (compile with `-arch=sm_75`).

Active learning gaps (concepts being built right now):
- Pointers and manual memory management (biggest gap from Java/Python)
- Host vs device memory model (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
- CUDA function qualifiers (`__global__`, `__device__`, `__host__`)
- Kernel launch syntax `<<<blocks, threads>>>` and index calculation
- C++ `new` / `delete[]` vs Python garbage collection
 
## Architecture Decisions
 
- All core math written from scratch; no cuBLAS/cuDNN. (Week 0)
- Validate every kernel against a PyTorch reference. (Week 0)
- fp16 storage / fp32 accumulation as the starting convention. (Week 0)
- _(Add decisions here as they're made, with the week noted.)_
## What I've Tried (don't re-suggest)
 
- Nothing yet — project start, Week 1.
- _(Log completed kernels and dead ends here each week so we don't repeat them.)_