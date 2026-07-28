- Doing sum reduction, array summed and return summed value
- Wrote CPU reference (reduction.cpp): serial int sum, validated by hand against closed-form sum (n=256, arr[i]=i*2+3 -> 66048)
- Wrote reduction_naive.cu: single block (numBlocks=1, threadsPerBlock=256), shared-memory interleaved-addressing tree reduction (Harris kernel 1 style, tid % (2*stride)==0 divergent branch). Validated against inline CPU float sum with fabsf tolerance on Colab T4 (sm_75). Passed.
- Profiled reduction_naive with ncu at n=256, Memory Workload Analysis section:

  | Metric              | Value        |
  |----------------------|-------------|
  | Memory Throughput    | 693.97 MB/s |
  | Mem Busy             | 0.42%       |
  | Max Bandwidth        | 0.33%       |
  | L1/TEX Hit Rate      | 0%          |
  | L2 Hit Rate          | 78.67%      |
  | Mem Pipes Busy       | 0.26%       |

- Interpretation: all utilization numbers near zero because n=256 (1KB total) is far too small to stress the memory subsystem — kernel finishes before memory system ramps up, so total time is dominated by launch/sync overhead, not data movement. L1/TEX Hit Rate of 0% is structural, not a problem: each thread reads global input[gi] exactly once (no reuse to hit against), all further work happens in shared memory which this metric doesn't track. L2 Hit Rate of 78.67% reflects ordinary spatial locality from adjacent warps' coalesced reads landing on cache lines already pulled in by neighboring warps.
- Lesson (echoes week3 caveat about working set vs cache size): need a much larger n (tens of thousands+) to get a memory profile with actual signal. Naive-vs-optimized comparison at n=256 would be meaningless.
- TODO next: reduction_optimized.cu — fix divergent branching + shared-memory bank conflicts from the naive kernel, scale n up substantially, add multi-block atomicAdd combine so results are comparable at a size where memory metrics actually move.
- Ran into a real bug scaling up: filled the 1M-element test array with i*3+1 and got "Failed" against the CPU reference. Root cause was float32 precision, not kernel logic — with n=1,048,576 the running sum climbs into the trillions, and float32 only holds ~7 accurate digits, so small per-element contributions get rounded away, differently depending on summation order (CPU serial vs GPU tree). Simulated both orders in Python: serial sum was off from the true (float64) answer by ~477 million, while the tree-order sum was off by only ~17,000 — confirms tree/pairwise summation is meaningfully more numerically accurate than naive serial summation at this scale, not just faster. Fix for now: filled h_in with 1.0f instead (sum stays at exactly n, well within float32's exact range) to keep the correctness check trustworthy while still using a large n for profiling. Proper fix (double-precision reference, relative tolerance) deferred to Week 5/8.
- Wrote reduction_optimized.cu: same multi-block/atomicAdd structure as naive, but sequential addressing in the tree loop (stride starts at blockDim.x/2, halves down to 1; `if (tid < stride)` instead of the naive `tid % (2*stride) == 0`) — fixes both warp divergence (active threads are now a contiguous low-tid block instead of a scattered modulo pattern) and shared-memory bank conflicts (consecutive tid now maps to consecutive addresses instead of a strided pattern). Passed.
- Profiled naive vs optimized at matched config (n=1,048,576, numBlocks=4096, threadsPerBlock=256), Memory Workload Analysis:

  | Metric              | Naive       | Optimized   |
  |----------------------|------------|-------------|
  | Memory Throughput    | 37.44 GB/s | 60.83 GB/s  |
  | Mem Busy             | 25.83%     | 41.13%      |
  | Max Bandwidth        | 45.97%     | 73.18%      |
  | L1/TEX Hit Rate      | 3.00%      | 0%          |
  | L2 Hit Rate          | 5.44%      | 4.99%       |
  | Mem Pipes Busy       | 45.97%     | 73.18%      |

- Headline result: fixing divergent branching + bank conflicts alone gave a 1.62x memory throughput improvement (37.44 -> 60.83 GB/s) and pushed Max Bandwidth utilization from 45.97% to 73.18%, at identical problem size. Mechanism: naive's divergent branch and bank conflicts force partial serialization within warps/shared-memory transactions each round, so the SM spends more time stalled and less time actually moving data; sequential addressing keeps whole warps uniformly active or idle at each step, reducing stalls and keeping the memory pipeline fed more continuously. L1/TEX Hit Rate difference (3.00% vs 0%) is small enough to be run-to-run noise, not attributed to the addressing change.
- Week 4 milestone met: correct sum reduction (naive + optimized) with one measured, profiler-backed optimization pass.