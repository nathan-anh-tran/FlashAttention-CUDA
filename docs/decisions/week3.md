- Implemented tiled matmul with TILE_WIDTH=16 shared-memory tiles, matching the 16x16 block dims from naive
- Got distracted with more FIFA World Cup
- Installed and ran Nsight Compute (ncu) on Colab T4 GPU
- Validated tiled kernel against CPU reference: passed
- Profiled naive vs tiled on m=512, k=1024, n=512, compared Memory Workload Analysis section:

  | Metric                    | Naive     | Tiled    |
  |---------------------------|-----------|----------|
  | Memory Throughput (DRAM)  | 57.95 GB/s| 6.82 GB/s|
  | L1/TEX Hit Rate           | 87.43%    | 3.35%    |
  | L2 Hit Rate               | 50.14%    | 97.15%   |

- Headline result: tiling cut actual DRAM traffic ~8.5x (57.95 -> 6.82 GB/s). Matches theory — TILE_WIDTH=16 predicts up to 16x fewer redundant global loads; 8.5x is a reasonable real-world number after boundary tiles/occupancy overhead.
- Non-obvious finding: L1 hit rate went DOWN under tiling (87% -> 3%), which looks backwards but isn't a regression. Naive's high L1 hit rate comes from the access pattern itself — threads sharing a row of A request the same address in the same cycle (a broadcast the cache serves cheaply), so a lot of *accidental* hardware-driven reuse happens. But naive still issues one independent load per thread per iteration, and once B's footprint (2MB) exceeds L1 capacity, most of that traffic falls through to L2, and only half of those hit (50% L2 hit rate) — the rest goes to DRAM. Tiled has almost no L1 hits because there's barely any redundant *request* left to hit against: each element is fetched from global memory exactly once per block, and the reuse happens explicitly via shared memory instead of the cache. Lesson: hit rate % alone doesn't capture absolute traffic volume — naive hits its cache more often per-request but issues far more requests overall.
- Caveat noted: working set (~4MB for A+B) is close to the T4's L2 cache size, so this comparison may partly reflect L2 catching naive's redundant traffic rather than a pure DRAM-bandwidth story. Worth re-running at larger matrix sizes (e.g. 2048/4096) to confirm the gap holds once the working set exceeds L2.
- TODO next: re-profile at larger sizes to rule out the L2-caching caveat above; write naive vs tiled numbers into benchmarks/results/ as CSV
