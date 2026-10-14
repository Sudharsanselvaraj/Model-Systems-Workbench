"""Report generator for inference serving benchmark simulations and cost crossover analyses."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from msu_llm_lab.serving.models import CostCrossoverComparison, ServingBenchmarkReport


class ServingReportGenerator:
    """Generates executive Markdown reports and tabular data for hardware and serving evaluations."""

    @staticmethod
    def generate_markdown(
        benchmark_report: ServingBenchmarkReport,
        cost_comparison: Optional[CostCrossoverComparison] = None,
    ) -> str:
        """Formats serving simulation findings and cost tradeoff into clean Markdown."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        sizing = benchmark_report.sizing

        md = [
            f"# MSU LLM Workbench — Self-Hosted vLLM Serving & Concurrency Report",
            f"",
            f"**Generated**: `{now_str}`  ",
            f"**Model Candidate**: `{benchmark_report.model_id}` ({benchmark_report.precision.value})  ",
            f"**Hardware Cluster**: `{benchmark_report.tensor_parallel_size}x {benchmark_report.hardware_name}`  ",
            f"**Max Recommended Concurrency**: `{benchmark_report.max_recommended_concurrency} workers`  ",
            f"**Peak Throughput**: `{benchmark_report.peak_throughput_tokens_sec:.1f} tokens/s`  ",
            f"",
            f"---",
            f"",
            f"## 1. Hardware VRAM Allocation Breakdown",
            f"",
            f"| Memory Component | Allocation (GB) | % of Total VRAM | Status / Role |",
            f"|---|---|---|---|",
            f"| **Model Weights** | `{sizing.weights_memory_gb:.2f} GB` | `{sizing.weights_memory_gb / max(0.1, sizing.total_gpu_vram_gb) * 100:.1f}%` | Quantized parameters in GPU HBM |",
            f"| **CUDA Runtime** | `{sizing.cuda_runtime_overhead_gb:.2f} GB` | `{sizing.cuda_runtime_overhead_gb / max(0.1, sizing.total_gpu_vram_gb) * 100:.1f}%` | PyTorch context and NCCL buffers |",
            f"| **Activation Buffer** | `{sizing.activation_overhead_gb:.2f} GB` | `{sizing.activation_overhead_gb / max(0.1, sizing.total_gpu_vram_gb) * 100:.1f}%` | Dynamic forward-pass working memory |",
            f"| **KV-Cache (PagedAttention)** | `{sizing.kv_cache_allocated_gb:.2f} GB` | `{sizing.kv_cache_allocated_gb / max(0.1, sizing.total_gpu_vram_gb) * 100:.1f}%` | Available pool for concurrent client sessions |",
            f"| **Total Cluster Memory** | `{sizing.total_gpu_vram_gb:.2f} GB` | `100.0%` | Deployable: `{'YES' if sizing.is_deployable else 'NO (OOM)'}` |",
            f"",
            f"> [!NOTE]",
            f"> KV-Cache consumption per token: **{sizing.kv_cache_per_token_bytes:.1f} bytes**. Supports up to **{sizing.max_concurrent_sessions_8k}** concurrent 8k student inquiry sessions.",
            f"",
            f"---",
            f"",
            f"## 2. Progressive Concurrency Ramp",
            f"",
            f"| Concurrency | Throughput (tok/s) | Mean TTFT | P95 TTFT | Mean TPOT | P95 Latency | KV Util | Preempt % | SLA Adherence |",
            f"|---|---|---|---|---|---|---|---|---|",
        ]

        for step in benchmark_report.concurrency_ramp:
            md.append(
                f"| **{step.concurrency_level}** | `{step.output_tokens_per_second:.1f}` | `{step.mean_ttft_ms:.1f}ms` | `{step.p95_ttft_ms:.1f}ms` | `{step.mean_tpot_ms:.2f}ms` | `{step.p95_total_latency_ms:.1f}ms` | `{step.kv_cache_utilization_pct:.1f}%` | `{step.preemption_rate_pct:.1f}%` | **{step.sla_adherence_pct:.1f}%** |"
            )

        if cost_comparison:
            md.extend([
                f"",
                f"---",
                f"",
                f"## 3. Cost-Crossover & Economic Viability Analysis",
                f"",
                f"- **Dedicated GPU Hosting**: `${cost_comparison.monthly_hosting_cost_usd:,.2f}/month` (`${cost_comparison.gpu_hourly_cost_usd:.2f}/hr` cluster rate)",
                f"- **Cloud API Cost**: `${cost_comparison.cost_per_query_cloud_usd:.6f} / student inquiry` ({cost_comparison.avg_prompt_tokens} prompt / {cost_comparison.avg_completion_tokens} completion tokens)",
                f"- **Break-Even Threshold**: **{cost_comparison.break_even_monthly_queries:,} queries/month** ({cost_comparison.break_even_daily_queries:,} queries/day)",
                f"- **Economic Feasibility for MSU**: `{'VIABLE' if cost_comparison.is_economically_viable_for_msu else 'UNJUSTIFIED'}`",
                f"",
                f"### Strategic Recommendation",
                f"> {cost_comparison.recommendation_summary}",
            ])

        return "\n".join(md)

    @classmethod
    def save_report(
        cls,
        benchmark_report: ServingBenchmarkReport,
        cost_comparison: Optional[CostCrossoverComparison] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Saves generated report to disk."""
        target_dir = output_dir or Path("reports")
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_model = benchmark_report.model_id.replace("/", "_")
        path = target_dir / f"serving_benchmark_{safe_model}_{ts}.md"

        content = cls.generate_markdown(benchmark_report, cost_comparison)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
