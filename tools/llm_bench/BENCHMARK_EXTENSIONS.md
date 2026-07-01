# llm_bench extensions for Qwen3 / Qwen3.5 TTFT benchmarking

This document describes a set of additive changes to `tools/llm_bench` used for
first-token-latency (TTFT) benchmarking of Qwen3 (dense) and Qwen3.5 (hybrid/SSM)
models across CPU / GPU / NPU. All changes are backward-compatible: existing
invocations behave as before unless the new flags are passed.

## Summary of changes

| Area | Files | What it adds |
|---|---|---|
| VLM prompt permutation | `task/visual_language_generation.py` | Per-iteration prompt mutation for the `visual_text_gen` task, on by default, to defeat the prefix cache so every measured iteration is a real cold prefill |
| Whole-run resource monitor | `llm_bench_utils/resource_monitor.py` (new), `benchmark.py` | A lightweight background sampler for CPU / per-core CPU / GPU / GPU-memory / NPU utilization and RSS/system memory over the whole run |
| Structured CSV/JSON output | `llm_bench_utils/output_csv.py`, `llm_bench_utils/output_json.py`, `benchmark.py` | `--benchmark_csv` / `--benchmark_json` writers with per-iteration latency breakdown (incl. pure-inference first-token latency and a `result_md5` of the decoded output) plus the run-level resource summary |
| NPU reranker static shape | `llm_bench_utils/ov_utils.py` | Force `pad_to_max_length` on NPU (which compiles static-shape IR only); CPU/GPU keep dynamic shapes |
| Docs | `README.md` | Documents the permutation behavior for both `text_gen` and `visual_text_gen` and the `--disable_prompt_permutation` flag |

## 1. VLM prompt permutation (`visual_text_gen`)

The existing `text_gen` task already permutes the prompt per iteration (by mutating
an early input token id) so that first-token latency reflects a cold prefill rather
than a prefix-cache hit. `visual_text_gen` receives a **string** prompt (no
`TokenizedInputs`), so token-id mutation is not possible. The new
`permute_vlm_prompt()` instead **prepends the iteration index** to the prompt string,
which changes the first token and misses the KV prefix cache from position 0.

- Enabled by default; disable with `--disable_prompt_permutation` for a cache-warm
  run with identical prompts every iteration.
- Consequence: with permutation on, decoded outputs (and their `result_md5`) may
  legitimately differ across iterations (the prompt differs); with it off, all
  iterations share one prompt and yield a stable `result_md5`. When comparing an
  optimization A/B, compare the **multiset** of per-iteration md5s, not a single value.

## 2. Whole-run resource monitor

`ResourceMonitor` (in `llm_bench_utils/resource_monitor.py`) runs a background thread
sampling, at a fixed interval:

- process CPU % (and per-core CPU %),
- GPU utilization and GPU memory utilization via `xpu-smi` (blank if `xpu-smi` is
  unavailable or does not expose compute utilization),
- NPU busy % (read from the NPU sysfs busy counter when present),
- process RSS and system memory (MiB).

It records a compact **run-level** summary (averages + peaks), intentionally separate
from the existing per-iteration memory-consumption collector so it does not add many
per-iteration columns.

New `benchmark.py` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--resource_monitor` | off | Enable the background resource monitor |
| `--resource_monitor_interval` | 500 | Sampling interval in milliseconds (floored at 100) |
| `--resource_gpu_command` | auto | Override the `xpu-smi` command line (also read from `XPU_SMI_CMD`) |

## 3. Structured CSV/JSON output

New `benchmark.py` flags `--benchmark_csv <path>` and `--benchmark_json <path>` write
a machine-readable report:

- Per-iteration latency breakdown, including **pure-inference first-token latency**
  (`1st_infer_latency_avg` / `first_token_infer_*`) separated from end-to-end TTFT,
  and a `result_md5` of the decoded output text for correctness/regression checks.
- The run-level resource summary from the monitor: `resource_cpu_avg_util`,
  per-core `resource_cpu_cpuN_avg_util`, `resource_gpu_avg_util`,
  `resource_gpu_memory_avg_util`, `resource_npu_avg_util`, and average/peak RSS and
  system memory.

These fields are what the TTFT analysis (PA vs SDPA, per-backend prefill breakdown)
consumes. `result_md5` is the primary correctness signal: under greedy decoding it is
a deterministic fingerprint of the full decoded sequence, so an optimization that is
numerically lossless leaves the md5 (multiset, when permutation is on) unchanged.

## 4. NPU reranker static shape

`ov_utils.py`: NPU compiles static-shape IR only. A statically-reshaped reranker model
expects every inference to feed exactly `[num_candidates, max_length]`, so padding is
forced to `max_length` on NPU (`config.pad_to_max_length = True`). CPU and GPU keep the
default dynamic-shape behavior (no forced padding).

## Example

```bash
# Qwen3.5 VLM, GPU, PA backend, cold-prefill TTFT with resource + structured output
python benchmark.py -m <model_dir> --task visual_text_gen \
  -p "What is openvino?" -n 8 -ic 128 \
  --resource_monitor --resource_monitor_interval 1000 \
  --benchmark_csv out.csv --benchmark_json out.json

# cache-warm variant (identical prompt each iteration, stable result_md5)
python benchmark.py -m <model_dir> --task visual_text_gen \
  -p "What is openvino?" -n 8 -ic 128 --disable_prompt_permutation
```
