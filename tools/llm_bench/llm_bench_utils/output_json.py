# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from llm_bench_utils.memory_monitor import MemoryUnit, MemThreadHandler


def estimate_throughput(latency, bs, ms=True):
    # latency may arrive as "" (e.g. text_embed / text_rerank have no 2nd-token
    # latency, so _mean() yields an empty string); coerce defensively.
    try:
        latency = float(latency)
        if latency > 0:
            factor = 1000.0 if ms else 1.0
            return round(int(bs) * factor / latency, 5)
    except (TypeError, ValueError):
        return None
    return None


def estimate_total_throughput(output_size, generation_time):
    try:
        if generation_time > 0:
            return round(float(output_size) / float(generation_time), 5)
    except (TypeError, ValueError):
        return None
    return None


def get_resource_usage_data(model_args):
    usage = model_args.get("resource_usage", {}) or {}
    def fmt(value):
        return round(value, 5) if isinstance(value, (int, float)) else value
    per_core = usage.get("cpu_per_core_avg_util_percent", {}) or {}
    return {
        "resource_cpu_avg_util_percent": fmt(usage.get("cpu_avg_util_percent", "")),
        "resource_cpu_per_core_avg_util_percent": {key: fmt(value) for key, value in per_core.items()},
        "resource_gpu_avg_util_percent": fmt(usage.get("gpu_avg_util_percent", "")),
        "resource_gpu_memory_avg_util_percent": fmt(usage.get("gpu_memory_avg_util_percent", "")),
        "resource_npu_avg_util_percent": fmt(usage.get("npu_avg_util_percent", "")),
        "resource_avg_rss_mem_mib": fmt(usage.get("avg_rss_mem_mib", "")),
        "resource_peak_rss_mem_mib": fmt(usage.get("peak_rss_mem_mib", "")),
        "resource_avg_sys_mem_mib": fmt(usage.get("avg_sys_mem_mib", "")),
        "resource_peak_sys_mem_mib": fmt(usage.get("peak_sys_mem_mib", "")),
        "resource_memory_samples": usage.get("memory_samples", ""),
    }


def get_resource_cpu_core_csv_headers(model_args):
    resource_data = get_resource_usage_data(model_args)
    per_core = resource_data.get("resource_cpu_per_core_avg_util_percent", {}) or {}
    try:
        core_names = sorted(per_core.keys(), key=lambda name: int(name.replace("cpu", "")))
    except ValueError:
        core_names = sorted(per_core.keys())
    return [f"resource_cpu_{core}_avg_util(%)" for core in core_names]


def add_resource_usage_to_row(row, model_args):
    resource_data = get_resource_usage_data(model_args)
    row["resource_cpu_avg_util(%)"] = resource_data["resource_cpu_avg_util_percent"]
    for core, value in resource_data.get("resource_cpu_per_core_avg_util_percent", {}).items():
        row[f"resource_cpu_{core}_avg_util(%)"] = value
    row["resource_gpu_avg_util(%)"] = resource_data["resource_gpu_avg_util_percent"]
    row["resource_gpu_memory_avg_util(%)"] = resource_data["resource_gpu_memory_avg_util_percent"]
    row["resource_npu_avg_util(%)"] = resource_data["resource_npu_avg_util_percent"]
    row["resource_avg_rss_mem(MiB)"] = resource_data["resource_avg_rss_mem_mib"]
    row["resource_peak_rss_mem(MiB)"] = resource_data["resource_peak_rss_mem_mib"]
    row["resource_avg_sys_mem(MiB)"] = resource_data["resource_avg_sys_mem_mib"]
    row["resource_peak_sys_mem(MiB)"] = resource_data["resource_peak_sys_mem_mib"]
    row["resource_memory_samples"] = resource_data["resource_memory_samples"]


def _mean(values):
    filtered = []
    for value in values:
        if value == "" or value is None:
            continue
        if isinstance(value, (int, float)) and value < 0:
            continue
        try:
            filtered.append(float(value))
        except (TypeError, ValueError):
            continue
    return round(sum(filtered) / len(filtered), 5) if filtered else ""


def _first_value(values):
    for value in values:
        if value != "" and value is not None:
            return value
    return ""


def build_benchmark_summary_entries(
    model,
    framework,
    device,
    model_args,
    iter_data_list,
    pretrain_time,
    model_precision,
    memory_data_collector=None,
):
    prompt_idxs = sorted({item["prompt_idx"] for item in iter_data_list})
    entries = []
    for prompt_idx in prompt_idxs:
        measured = [item for item in iter_data_list if item["prompt_idx"] == prompt_idx and item["iteration"] > 0]
        if not measured:
            continue

        generation_time = _mean(item.get("generation_time") for item in measured)
        output_size = _mean(item.get("output_size") for item in measured)
        latency = _mean(item.get("latency") for item in measured)
        first_latency = _mean(item.get("first_token_latency") for item in measured)
        second_latency = _mean(item.get("other_tokens_avg_latency") for item in measured)
        first_infer_latency = _mean(item.get("first_token_infer_latency") for item in measured)
        second_infer_latency = _mean(item.get("other_tokens_infer_avg_latency") for item in measured)
        tokenization_time = _mean(item.get("tokenization_time") for item in measured)
        detokenization_time = _mean(item.get("detokenization_time") for item in measured)
        total_throughputs = [estimate_total_throughput(item.get("output_size"), item.get("generation_time")) for item in measured]
        total_throughput = _mean(total_throughputs)
        second_token_throughput = estimate_throughput(second_latency, model_args["batch_size"])

        entry = {
            "model": model,
            "framework": framework,
            "device": device,
            "precision": model_precision,
            "num_beams": model_args["num_beams"],
            "batch_size": model_args["batch_size"],
            "prompt_idx": prompt_idx,
            "measured_iterations": len(measured),
            "pretrain_time": round(pretrain_time, 5),
            "input_size": _mean(item.get("input_size") for item in measured),
            "infer_count": _first_value(item.get("infer_count") for item in measured),
            "generation_time": generation_time,
            "output_size": output_size,
            "latency": latency,
            "first_latency": first_latency,
            "second_avg_latency": second_latency,
            "first_infer_latency": first_infer_latency,
            "second_infer_avg_latency": second_infer_latency,
            "second_token_throughput": second_token_throughput if second_token_throughput else "",
            "total_throughput": total_throughput,
            "tokenization_time": tokenization_time,
            "detokenization_time": detokenization_time,
            "result_md5": _first_value(item.get("result_md5") for item in measured),
        }
        entry.update(get_resource_usage_data(model_args))
        entry.update(get_pre_gen_memory_data(memory_data_collector))
        entries.append(entry)
    return entries


def append_benchmark_summary(
    report_file,
    model,
    framework,
    device,
    model_args,
    iter_data_list,
    pretrain_time,
    model_precision,
    memory_data_collector,
):
    entries = build_benchmark_summary_entries(
        model, framework, device, model_args, iter_data_list, pretrain_time, model_precision, memory_data_collector
    )
    out_file = Path(report_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out_file.exists() and out_file.stat().st_size > 0:
        try:
            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            existing = []
    out_file.write_text(json.dumps(existing + entries, indent=4), encoding="utf-8")


def write_result(report_file, model, framework, device, model_args, iter_data_list, pretrain_time, model_precision, iter_timestamp, memory_data_collector):
    metadata = {'model': model, 'framework': framework, 'device': device, 'precision': model_precision,
                'num_beams': model_args['num_beams'], 'batch_size': model_args['batch_size']}
    result = []
    total_iters = len(iter_data_list)
    for i in range(total_iters):
        iter_data = iter_data_list[i]
        latency = iter_data["latency"]
        generation_time = iter_data["generation_time"]
        first_latency = iter_data["first_token_latency"]
        other_latency = iter_data["other_tokens_avg_latency"]
        first_token_infer_latency = iter_data["first_token_infer_latency"]
        other_token_infer_latency = iter_data["other_tokens_infer_avg_latency"]

        # optional metrics
        tokenization_time = iter_data.get("tokenization_time")
        detokenization_time = iter_data.get("detokenization_time")
        max_rss_mem = iter_data.get("max_rss_mem_consumption")
        max_sys_mem = iter_data.get("max_sys_mem_consumption")
        rss_mem_increase = iter_data.get("max_rss_mem_increase")
        sys_mem_increase = iter_data.get("max_sys_mem_increase")
        rss_mem_share = iter_data.get("max_rss_mem_share")
        sys_mem_share = iter_data.get("max_sys_mem_share")

        result_md5 = []
        for idx_md5 in range(len(iter_data['result_md5'])):
            result_md5.append(iter_data['result_md5'][idx_md5])

        timestamp_start, timestamp_end = get_timestamp(iter_data['iteration'], iter_data['prompt_idx'], iter_timestamp)

        if first_token_infer_latency:
            first_token_infer_latency = round(first_token_infer_latency, 5)
        if other_token_infer_latency:
            other_token_infer_latency = round(other_token_infer_latency, 5)
        if tokenization_time:
            tokenization_time = round(tokenization_time, 5)
        if detokenization_time:
            detokenization_time = round(detokenization_time, 5)
        if generation_time:
            generation_time = round(generation_time, 5)
        total_throughput = estimate_total_throughput(iter_data["output_size"], generation_time)

        res_data = {
            "iteration": iter_data["iteration"],
            "input_size": iter_data["input_size"],
            "infer_count": iter_data["infer_count"],
            "output_size": iter_data["output_size"],
            "generation_time": generation_time,
            "latency": round(latency, 5) if latency != "" else latency,
            "first_latency": round(first_latency, 5) if first_latency != "" else first_latency,
            "second_avg_latency": round(other_latency, 5) if other_latency != "" else other_latency,
            "first_infer_latency": first_token_infer_latency,
            "second_infer_avg_latency": other_token_infer_latency,
            "tokenization_time": tokenization_time,
            "detokenization_time": detokenization_time,
            "prompt_idx": iter_data["prompt_idx"],
            "result_md5": result_md5,
            "start": timestamp_start,
            "end": timestamp_end,
        }

        if max_rss_mem:
            res_data["max_rss_mem"] = round(max_rss_mem, 5)
        if max_sys_mem:
            res_data["max_sys_mem"] = round(max_sys_mem, 5)
        if rss_mem_increase:
            res_data["max_increase_rss_mem"] = round(rss_mem_increase, 5)
        if sys_mem_increase:
            res_data["max_increase_sys_mem"] = round(sys_mem_increase, 5)
        if rss_mem_share:
            res_data["max_share_rss_mem"] = round(rss_mem_share, 3)
        if sys_mem_share:
            res_data["max_share_sys_mem"] = round(sys_mem_share, 3)

        second_token_throughput = estimate_throughput(other_latency, model_args["batch_size"])
        if second_token_throughput:
            res_data["second_token_throughput"] = second_token_throughput
        if total_throughput:
            res_data["total_throughput"] = total_throughput
        result.append(res_data)

    keys_to_average = [
        'generation_time',
        'latency',
        'first_latency',
        'second_avg_latency',
        'first_infer_latency',
        'second_infer_avg_latency',
        'tokenization_time',
        'detokenization_time'
    ]
    results_averaged = {}
    for key in keys_to_average:
        values = [x[key] for x in result[1:] if x[key] != '']
        if len(values) > 0:
            results_averaged[key] = round(sum(values) / len(values), 5)

    if "second_avg_latency" in results_averaged:
        avg_2nd_tokens_latency = results_averaged.get("second_avg_latency")
        second_token_throughput = estimate_throughput(avg_2nd_tokens_latency, model_args["batch_size"])
        if second_token_throughput:
            results_averaged["second_token_throughput"] = second_token_throughput
    if len(result) > 1:
        throughputs = [x.get("total_throughput") for x in result[1:] if x.get("total_throughput") is not None]
        if throughputs:
            results_averaged["total_throughput"] = round(sum(throughputs) / len(throughputs), 5)

    output_result = {'metadata': metadata,
                     'perfdata': {'compile_time': pretrain_time,
                                  'results': result,
                                  'resource_usage': model_args.get("resource_usage", {})} | get_pre_gen_memory_data(memory_data_collector)}
    if len(results_averaged) > 0:
        output_result['perfdata']['results_averaged'] = results_averaged

    with open(report_file, 'w') as outfile:
        json.dump(output_result, outfile, indent=4)


def get_pre_gen_memory_data(memory_data_collector: MemThreadHandler | None, print_unit: MemoryUnit | None = None):
    no_info = MemThreadHandler.MEMORY_NOT_COLLECTED
    suffix = f"({MemThreadHandler.DEF_MEM_UNIT.value})"
    if print_unit is not None:
        suffix = f"({print_unit.value})"
    data = {
        f"initial_sys_mem{suffix}": no_info,
        f"initial_rss_mem{suffix}": no_info,
        f"compile_max_rss_mem{suffix}": no_info,
        f"compile_max_sys_mem{suffix}": no_info,
        f"compile_max_increase_rss_mem{suffix}": no_info,
        f"compile_max_increase_sys_mem{suffix}": no_info,
    }
    if not memory_data_collector:
        return data

    data = data | memory_data_collector.get_initial_mem_data(print_unit)
    data = data | memory_data_collector.get_compilation_mem_data(print_unit)
    return data


def get_timestamp(iter_idx, prompt_idx, iter_timestamp):
    timestamp_start = ''
    timestamp_end = ''
    if iter_idx in iter_timestamp.keys():
        if prompt_idx in iter_timestamp[iter_idx].keys():
            timestamp_start = iter_timestamp[iter_idx][prompt_idx]['start']
            timestamp_end = iter_timestamp[iter_idx][prompt_idx]['end']

    return timestamp_start, timestamp_end
