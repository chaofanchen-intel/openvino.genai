# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shlex
import subprocess
import threading
import time

import psutil


class ResourceMonitor:
    """Lightweight whole-run CPU/GPU/NPU utilization and memory sampler.

    This monitor is intentionally independent from the existing
    memory_consumption collector. It records a compact run-level summary for
    CSV/JSON reports instead of adding many per-iteration memory columns.
    """

    def __init__(self, enabled=True, interval_ms=500, gpu_command=None):
        self.enabled = enabled
        self.interval_s = max(100, int(interval_ms)) / 1000.0
        self._gpu_available = self._has_command("xpu-smi")
        default_gpu_command = self._default_gpu_command() if self._gpu_available else None
        self.gpu_command = gpu_command or os.environ.get("XPU_SMI_CMD") or default_gpu_command
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        self._cpu_samples = []
        self._cpu_per_core_samples = []
        self._gpu_samples = []
        self._gpu_memory_samples = []
        self._npu_samples = []
        self._rss_samples_mib = []
        self._sys_samples_mib = []
        self._npu_busy_path = self._find_npu_busy_path()
        self._prev_npu_busy_us = None
        self._prev_npu_time = None

    def start(self):
        if not self.enabled:
            return
        self._running = True
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        self._prev_npu_busy_us = self._read_npu_busy_time()
        self._prev_npu_time = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def summary(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "cpu_avg_util_percent": self._avg(self._cpu_samples),
                "cpu_per_core_avg_util_percent": self._avg_per_core(self._cpu_per_core_samples),
                "cpu_samples": len(self._cpu_samples),
                "gpu_available": self._gpu_available,
                "gpu_avg_util_percent": self._avg(self._gpu_samples),
                "gpu_memory_avg_util_percent": self._avg(self._gpu_memory_samples),
                "gpu_samples": len(self._gpu_samples),
                "gpu_memory_samples": len(self._gpu_memory_samples),
                "npu_available": self._npu_busy_path is not None,
                "npu_avg_util_percent": self._avg(self._npu_samples),
                "npu_samples": len(self._npu_samples),
                "avg_rss_mem_mib": self._avg(self._rss_samples_mib),
                "peak_rss_mem_mib": max(self._rss_samples_mib) if self._rss_samples_mib else "",
                "avg_sys_mem_mib": self._avg(self._sys_samples_mib),
                "peak_sys_mem_mib": max(self._sys_samples_mib) if self._sys_samples_mib else "",
                "memory_samples": len(self._rss_samples_mib),
            }

    def _loop(self):
        while self._running:
            self._sample_once()
            time.sleep(self.interval_s)

    def _sample_once(self):
        cpu = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        rss = self._rss_mib()
        sys_mem = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 * 1024)
        gpu, gpu_memory = self._read_gpu_stats()
        npu = self._read_npu_util()
        with self._lock:
            self._cpu_samples.append(cpu)
            self._cpu_per_core_samples.append(cpu_per_core)
            self._rss_samples_mib.append(rss)
            self._sys_samples_mib.append(sys_mem)
            if gpu is not None:
                self._gpu_samples.append(gpu)
            if gpu_memory is not None:
                self._gpu_memory_samples.append(gpu_memory)
            if npu is not None:
                self._npu_samples.append(npu)

    def _rss_mib(self):
        rss = self._process.memory_info().rss
        try:
            for child in self._process.children(recursive=True):
                rss += child.memory_info().rss
        except (psutil.Error, OSError):
            pass
        return rss / (1024 * 1024)

    def _read_gpu_stats(self):
        if not self._gpu_available or not self.gpu_command:
            return None, None
        try:
            result = subprocess.run(
                shlex.split(self.gpu_command), capture_output=True, text=True, timeout=5
            )
        except (subprocess.TimeoutExpired, OSError):
            return None, None
        if result.returncode != 0:
            return None, None
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None, None
        gpu_values = []
        gpu_memory_values = []
        self._collect_gpu_values(data, gpu_values, gpu_memory_values)
        gpu_util = max(0.0, min(100.0, max(gpu_values))) if gpu_values else None
        gpu_memory_util = max(0.0, min(100.0, max(gpu_memory_values))) if gpu_memory_values else None
        return gpu_util, gpu_memory_util

    def _read_npu_util(self):
        busy = self._read_npu_busy_time()
        now = time.monotonic()
        util = None
        if busy is not None and self._prev_npu_busy_us is not None and self._prev_npu_time is not None:
            delta_busy_s = (busy - self._prev_npu_busy_us) / 1_000_000.0
            delta_time_s = now - self._prev_npu_time
            if delta_time_s > 0:
                util = max(0.0, min(100.0, 100.0 * delta_busy_s / delta_time_s))
        self._prev_npu_busy_us = busy
        self._prev_npu_time = now
        return util

    def _read_npu_busy_time(self):
        if self._npu_busy_path is None:
            return None
        try:
            with open(self._npu_busy_path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except PermissionError:
            try:
                result = subprocess.run(
                    ["sudo", "-n", "cat", self._npu_busy_path], capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    return int(result.stdout.strip())
            except (subprocess.TimeoutExpired, OSError, ValueError):
                pass
        except (OSError, ValueError):
            pass
        return None

    @staticmethod
    def _collect_gpu_values(node, gpu_values, gpu_memory_values):
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if lowered in {"compute", "render"} and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and isinstance(item.get("value"), (int, float)):
                            gpu_values.append(float(item["value"]))
                    continue
                if isinstance(value, (int, float)):
                    metrics_type = str(node.get("metrics_type", "")).lower()
                    if "memory_utilization" in metrics_type or lowered in {
                        "memory_util", "memory_utilization", "gpu_memory_util", "gpu_memory_utilization"
                    }:
                        gpu_memory_values.append(float(value))
                    elif "gpu_utilization" in metrics_type or "engine_util" in lowered:
                        gpu_values.append(float(value))
                    elif lowered in {"gpu_util", "gpu_utilization", "compute", "render"}:
                        gpu_values.append(float(value))
                else:
                    ResourceMonitor._collect_gpu_values(value, gpu_values, gpu_memory_values)
        elif isinstance(node, list):
            for item in node:
                ResourceMonitor._collect_gpu_values(item, gpu_values, gpu_memory_values)

    @staticmethod
    def _has_command(command):
        try:
            return subprocess.run(["which", command], capture_output=True, timeout=3).returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    @staticmethod
    def _find_npu_busy_path():
        driver_path = "/sys/bus/pci/drivers/intel_vpu"
        if not os.path.isdir(driver_path):
            return None
        for entry in os.listdir(driver_path):
            if not entry.startswith("0000:"):
                continue
            busy_path = os.path.join(driver_path, entry, "npu_busy_time_us")
            if os.path.exists(busy_path):
                return busy_path
        return None

    @staticmethod
    def _default_gpu_command():
        try:
            result = subprocess.run(
                ["xpu-smi", "discovery", "-j"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for item in data.get("device_list", []):
                    if str(item.get("device_type", "")).upper() == "GPU":
                        device_id = item.get("device_id")
                        if device_id is not None:
                            return f"sudo -n xpu-smi stats -d {device_id} -j"
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
            pass
        return "sudo -n xpu-smi stats -d 0 -j"

    @staticmethod
    def _avg(values):
        return sum(values) / len(values) if values else ""

    @staticmethod
    def _avg_per_core(samples):
        if not samples:
            return {}
        result = {}
        core_count = max(len(sample) for sample in samples)
        for idx in range(core_count):
            values = [sample[idx] for sample in samples if idx < len(sample)]
            if values:
                result[f"cpu{idx}"] = sum(values) / len(values)
        return result