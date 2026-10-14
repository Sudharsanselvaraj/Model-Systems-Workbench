"""Physics-based discrete-event inference serving and concurrency simulator."""

import math
from typing import List, Optional
from msu_llm_lab.serving.models import (
    ConcurrencyStepResult,
    GPUProfile,
    MemorySizingResult,
    ModelArchitecture,
    QuantizationPrecision,
    ServingBenchmarkReport,
)
from msu_llm_lab.serving.profiles import get_hardware_profile, get_model_architecture
from msu_llm_lab.serving.sizing import HardwareSizingEngine


class ServingBenchmarkSimulator:
    """Simulates continuous batching, prefill/decode phases, and progressive concurrency ramps."""

    def __init__(
        self,
        avg_prompt_tokens: int = 1200,
        avg_completion_tokens: int = 350,
        model_flops_utilization: float = 0.42,  # Typical MFU for vLLM PagedAttention
        bandwidth_efficiency: float = 0.75,     # Practical memory bus efficiency
    ):
        self.avg_prompt_tokens = avg_prompt_tokens
        self.avg_completion_tokens = avg_completion_tokens
        self.mfu = model_flops_utilization
        self.bwe = bandwidth_efficiency

    def simulate_ramp(
        self,
        model: ModelArchitecture,
        gpu: GPUProfile,
        precision: QuantizationPrecision = QuantizationPrecision.BF16,
        tensor_parallel_size: int = 1,
        concurrency_levels: Optional[List[int]] = None,
    ) -> ServingBenchmarkReport:
        """Executes full progressive concurrency benchmark simulation from 1 to 150 workers."""
        if concurrency_levels is None:
            concurrency_levels = [1, 5, 10, 25, 50, 100, 150]

        # 1. Run Hardware Sizing calculation
        sizing = HardwareSizingEngine.calculate_sizing(
            model=model,
            gpu=gpu,
            precision=precision,
            tensor_parallel_size=tensor_parallel_size,
        )

        bytes_per_param = HardwareSizingEngine.get_bytes_per_param(precision)
        total_bandwidth_gb_s = gpu.memory_bandwidth_gb_s * tensor_parallel_size * self.bwe
        total_tflops = gpu.fp16_tflops * tensor_parallel_size * self.mfu * 1e12

        # Model weights in bytes
        model_weight_bytes = model.param_count_billions * 1e9 * bytes_per_param

        # Baseline single-token decode time (memory bandwidth bound)
        # Time per token (seconds) = model_weight_bytes / memory_bandwidth_bytes_per_sec
        base_tpot_sec = model_weight_bytes / (total_bandwidth_gb_s * (1024 ** 3))
        base_tpot_ms = base_tpot_sec * 1000.0

        # Baseline prefill time (compute bound)
        # FLOPs = 2 * prompt_tokens * param_count
        prompt_flops = 2 * self.avg_prompt_tokens * (model.param_count_billions * 1e9)
        base_prefill_sec = prompt_flops / total_tflops
        base_ttft_ms = base_prefill_sec * 1000.0

        # Maximum concurrent active requests supported in KV cache buffer
        kv_slot_capacity = max(1, sizing.max_concurrent_sessions_8k)

        step_results: List[ConcurrencyStepResult] = []
        max_rec_concurrency = 1
        peak_tokens_sec = 0.0

        for c in concurrency_levels:
            if not sizing.is_deployable:
                # OOM configuration
                step_results.append(
                    ConcurrencyStepResult(
                        concurrency_level=c,
                        requests_per_second=0.0,
                        output_tokens_per_second=0.0,
                        mean_ttft_ms=9999.0,
                        p50_ttft_ms=9999.0,
                        p95_ttft_ms=9999.0,
                        p99_ttft_ms=9999.0,
                        mean_tpot_ms=999.0,
                        p50_total_latency_ms=9999.0,
                        p95_total_latency_ms=9999.0,
                        p99_total_latency_ms=9999.0,
                        kv_cache_utilization_pct=100.0,
                        preemption_rate_pct=100.0,
                        sla_adherence_pct=0.0,
                    )
                )
                continue

            # Batching dynamics: Continuous batching with PagedAttention
            # If concurrency <= capacity: healthy linear throughput scaling
            # If concurrency > capacity: requests must queue or preempt
            saturation_ratio = c / kv_slot_capacity

            if saturation_ratio <= 1.0:
                kv_util = min(100.0, (c / kv_slot_capacity) * 100.0)
                preemption_pct = 0.0
                queue_delay_ms = 0.0
            else:
                kv_util = 100.0
                excess = c - kv_slot_capacity
                preemption_pct = min(100.0, (excess / c) * 35.0)
                # Non-linear queuing delay under saturation
                queue_delay_ms = (saturation_ratio ** 1.8) * 450.0

            # Batch size impact on TPOT (vLLM amortizes weight memory fetch over active requests in batch)
            # Up to a threshold, batching increases output tokens/sec while modestly increasing individual TPOT
            effective_batch = min(c, kv_slot_capacity)
            batch_overhead = 1.0 + (0.018 * math.sqrt(effective_batch))
            step_tpot_ms = base_tpot_ms * batch_overhead

            # TTFT with chunked prefill interference and queueing
            mean_ttft = base_ttft_ms * (1.0 + (0.04 * effective_batch)) + queue_delay_ms
            p50_ttft = mean_ttft * 0.94
            p95_ttft = mean_ttft * 1.35
            p99_ttft = mean_ttft * 1.85

            # Total latency = TTFT + (completion_tokens * TPOT)
            generation_ms = self.avg_completion_tokens * step_tpot_ms
            mean_total = mean_ttft + generation_ms
            p50_total = p50_ttft + generation_ms
            p95_total = p95_ttft + (generation_ms * 1.15)
            p99_total = p99_ttft + (generation_ms * 1.35)

            # System Throughput
            # Output tokens/sec = (effective_batch * 1000.0) / step_tpot_ms
            out_tokens_sec = (effective_batch * 1000.0) / step_tpot_ms
            reqs_sec = out_tokens_sec / self.avg_completion_tokens

            if out_tokens_sec > peak_tokens_sec:
                peak_tokens_sec = out_tokens_sec

            # SLA Adherence (<1.5s TTFT and <3.5s total latency)
            if p95_ttft <= 1500.0 and p95_total <= 3500.0:
                sla_pct = 100.0 - (preemption_pct * 0.5)
            elif p50_total <= 3500.0:
                penalty = (p95_total - 3500.0) / 100.0
                sla_pct = max(0.0, 95.0 - penalty - preemption_pct)
            else:
                sla_pct = max(0.0, 50.0 - ((mean_total - 3500.0) / 50.0))

            sla_pct = round(max(0.0, min(100.0, sla_pct)), 1)

            if sla_pct >= 95.0 and c > max_rec_concurrency:
                max_rec_concurrency = c

            step_results.append(
                ConcurrencyStepResult(
                    concurrency_level=c,
                    requests_per_second=round(reqs_sec, 2),
                    output_tokens_per_second=round(out_tokens_sec, 1),
                    mean_ttft_ms=round(mean_ttft, 1),
                    p50_ttft_ms=round(p50_ttft, 1),
                    p95_ttft_ms=round(p95_ttft, 1),
                    p99_ttft_ms=round(p99_ttft, 1),
                    mean_tpot_ms=round(step_tpot_ms, 2),
                    p50_total_latency_ms=round(p50_total, 1),
                    p95_total_latency_ms=round(p95_total, 1),
                    p99_total_latency_ms=round(p99_total, 1),
                    kv_cache_utilization_pct=round(kv_util, 1),
                    preemption_rate_pct=round(preemption_pct, 1),
                    sla_adherence_pct=sla_pct,
                )
            )

        notes = (
            f"Evaluated {model.display_name} on {tensor_parallel_size}x {gpu.name} ({precision.value}). "
            f"Peak throughput: {peak_tokens_sec:.1f} tokens/s. Recommended concurrency ceiling: {max_rec_concurrency} workers."
        )

        return ServingBenchmarkReport(
            model_id=model.model_id,
            precision=precision,
            hardware_name=gpu.name,
            tensor_parallel_size=tensor_parallel_size,
            sizing=sizing,
            concurrency_ramp=step_results,
            max_recommended_concurrency=max_rec_concurrency,
            peak_throughput_tokens_sec=round(peak_tokens_sec, 1),
            notes=notes,
        )
