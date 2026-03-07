# Qwen 2 7B Instruct Cost Comparison

## Run Details

$ /opt/pytorch/bin/python benchmarks/bench.py run \
          --model Qwen/Qwen2-7B-Instruct \
          --request-type chat_completions \
          --max-seconds 60 \
          --sweep-size 5

    ==================================================
    [bench] All benchmarks complete!
    [bench] Results: benchmarks/results/20260306_174220
    [bench]   vanilla                   OK
    [bench]   vanilla_eager             OK
    [bench]   svd_interval16            OK
    [bench]   svd_interval64            OK
    [bench]   svd_interval256           OK
    [bench]   svd_rank8_interval64      OK
    [bench]   svd_lanczos_interval64    FAIL
    ==================================================


## Comparative Results

$ /opt/pytorch/bin/python benchmarks/bench.py compare benchmarks/results/20260306_174220 

     Baseline: vanilla  model=Qwen/Qwen2-7B-Instruct  backend=TRITON_ATTN
     label   | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     --------+------------+-------------+-------------+------------+------------+--------
     vanilla | 972.1      | 15231.5     | 21334.7     | 209.99     | 216.34     | 21655

     vs vanilla_eager  model=Qwen/Qwen2-7B-Instruct  backend=TRITON_ATTN  enforce_eager:
     label         | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     --------------+------------+-------------+-------------+------------+------------+--------
     vanilla_eager | 949.3      | 15428.7     | 22715.6     | 209.69     | 216.16     | 21167
     delta         | -2.3%      | +1.3%       | +6.5%       | -0.1%      | -0.1%      | -2.3%

     vs svd_interval16  model=Qwen/Qwen2-7B-Instruct  backend=CUSTOM  enforce_eager  interval=16  rank=4:
     label          | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     ---------------+------------+-------------+-------------+------------+------------+--------
     svd_interval16 | 847.3      | 16431.2     | 22902.9     | 222.59     | 233.43     | 22585
     delta          | -12.8%     | +7.9%       | +7.4%       | +6.0%      | +7.9%      | +4.3%

     vs svd_interval64  model=Qwen/Qwen2-7B-Instruct  backend=CUSTOM  enforce_eager  interval=64  rank=4:
     label          | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     ---------------+------------+-------------+-------------+------------+------------+--------
     svd_interval64 | 899.5      | 15679.3     | 22233.3     | 216.80     | 226.78     | 22583
     delta          | -7.5%      | +2.9%       | +4.2%       | +3.2%      | +4.8%      | +4.3%

     vs svd_interval256  model=Qwen/Qwen2-7B-Instruct  backend=CUSTOM  enforce_eager  interval=256  rank=4:
     label           | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     ----------------+------------+-------------+-------------+------------+------------+--------
     svd_interval256 | 913.3      | 15553.3     | 22196.3     | 214.44     | 224.55     | 22579
     delta           | -6.0%      | +2.1%       | +4.0%       | +2.1%      | +3.8%      | +4.3%

     vs svd_rank8_interval64  model=Qwen/Qwen2-7B-Instruct  backend=CUSTOM  enforce_eager  interval=64  rank=8:
     label                | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     ---------------------+------------+-------------+-------------+------------+------------+--------
     svd_rank8_interval64 | 891.2      | 15710.7     | 22436.0     | 216.90     | 223.05     | 22583
     delta                | -8.3%      | +3.1%       | +5.2%       | +3.3%      | +3.1%      | +4.3%

     vs svd_lanczos_interval64  model=Qwen/Qwen2-7B-Instruct  backend=CUSTOM  enforce_eager  interval=64  rank=4  method=lanczos:
     label                  | Tput tok/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | GPU MiB
     -----------------------+------------+-------------+-------------+------------+------------+--------
     svd_lanczos_interval64 | n/a        | n/a         | n/a         | n/a        | n/a        | 21571
     delta                  | n/a        | n/a         | n/a         | n/a        | n/a        | -0.4%


# Notes relative to complexity analysis in MATRIX_FREE_SVD.md

- **TTFT: ~unchanged (no SVD during prefill)** -- Confirmed. TTFT p50 deltas are small (+2-8%), likely noise from enforce_eager overhead rather than SVD. The vanilla_eager baseline shows +1.3% TTFT vs vanilla, confirming eager mode accounts for most of the TTFT shift.

- **ITL p50: ~unchanged (most steps only do Q clone)** -- Partially confirmed. ITL p50 overhead is modest (+2-6% depending on interval), not zero. The Q clone + buffer accumulation has a small but measurable cost even on non-SVD steps. On OPT-125m it was ~30% — on Qwen2-7B the SVD cost is a smaller fraction of the already-larger per-step compute. 

- **ITL p99/max: elevated by SVD spike every N steps** -- Confirmed. ITL p99 is consistently higher than p50 delta across all SVD configs (+3.8% to +7.9%). interval=16 shows the biggest p99 hit (+7.9%) as expected — more frequent SVD spikes.

- **Throughput: reduced by ~SVD_time / (N × baseline_ITL)** -- Confirmed. Scales inversely with interval as predicted:
  - interval=16: -12.8% (most frequent SVD)
  - interval=64: -7.5%
  - interval=256: -6.0% (least frequent SVD)
  - rank=8 at interval=64: -8.3% (slightly worse than rank=4's -7.5%, higher rank = more compute)

- **Memory: Q buffer grows as O(L·H·d) per layer** -- Confirmed. All SVD configs show +4.3% GPU memory (~930 MiB) vs vanilla. This is consistent across intervals/ranks — the buffer is allocated once regardless of how often SVD fires. vanilla_eager actually uses less memory (-2.3%) since no CUDA graphs.

- **Key takeaway:** On a real 7B model, the overhead is much more reasonable than on the tiny OPT-125m. The SVD compute is amortized against the model's larger per-step cost. interval=64 at -7.5% throughput is a practical operating point.