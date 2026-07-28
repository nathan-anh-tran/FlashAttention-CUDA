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