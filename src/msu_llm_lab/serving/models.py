"""Data models for vLLM self-hosted inference serving, hardware sizing, and concurrency simulation."""

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuantizationPrecision(str, Enum):
    """Model weight and KV-cache precision formats."""
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    AWQ_INT4 = "AWQ_INT4"
    GPTQ_INT4 = "GPTQ_INT4"


class GPUProfile(BaseModel):
    """Specification of target GPU accelerator hardware."""
    name: str = Field(description="Commercial GPU name (e.g. 'NVIDIA A100-SXM4-80GB')")
    vram_gb: float = Field(gt=0, description="Available GPU memory in gigabytes")
    memory_bandwidth_gb_s: float = Field(gt=0, description="Peak memory bandwidth in GB/s")
    fp16_tflops: float = Field(gt=0, description="Half-precision tensor core TFLOPS")
    hourly_rate_usd: Decimal = Field(ge=0, description="Estimated hourly cloud rental rate (RunPod/Lambda)")
    interconnect: str = Field(default="PCIe 4.0", description="Bus or interconnect: NVLink, PCIe 4.0, or Unified")


class ModelArchitecture(BaseModel):
    """Transformer architecture specifications used for hardware sizing and memory modeling."""
    model_id: str
    display_name: str
    param_count_billions: float = Field(gt=0)
    hidden_size: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0, description="Number of Key-Value heads (equals attention heads for MHA, fewer for GQA)")
    vocab_size: int = Field(default=152064)
    default_context_window: int = Field(default=32768)
    supported_precisions: List[QuantizationPrecision] = Field(
        default_factory=lambda: [QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4]
    )

    @property
    def is_gqa(self) -> bool:
        """Indicates whether Grouped-Query Attention is used to reduce KV cache size."""
        return self.num_kv_heads < self.num_attention_heads


class MemorySizingResult(BaseModel):
    """Detailed memory breakdown for a model deployed on specific GPU topology."""
    model_id: str
    precision: QuantizationPrecision
    tensor_parallel_size: int = Field(ge=1, description="Number of GPUs pooled via Tensor Parallelism (TP)")
    gpu_profile: GPUProfile
    total_gpu_vram_gb: float
    weights_memory_gb: float
    kv_cache_per_token_bytes: float
    kv_cache_allocated_gb: float
    activation_overhead_gb: float
    cuda_runtime_overhead_gb: float
    total_memory_required_gb: float
    is_deployable: bool
    max_concurrent_sessions_8k: int = Field(description="Max concurrent 8k sessions accommodated in KV cache")
    max_concurrent_sessions_32k: int = Field(description="Max concurrent 32k sessions accommodated in KV cache")
    limiting_factor: str = Field(description="VRAM capacity, KV-cache starvation, or compute bounds")


class ConcurrencyStepResult(BaseModel):
    """Performance telemetry at a specific concurrent client load level."""
    concurrency_level: int = Field(ge=1, description="Number of concurrent student inquiry workers")
    requests_per_second: float = Field(ge=0)
    output_tokens_per_second: float = Field(ge=0)
    mean_ttft_ms: float = Field(ge=0, description="Mean Time-To-First-Token in ms (prefill phase)")
    p50_ttft_ms: float = Field(ge=0)
    p95_ttft_ms: float = Field(ge=0)
    p99_ttft_ms: float = Field(ge=0)
    mean_tpot_ms: float = Field(ge=0, description="Mean Time-Per-Output-Token in ms (decode phase)")
    p50_total_latency_ms: float = Field(ge=0)
    p95_total_latency_ms: float = Field(ge=0)
    p99_total_latency_ms: float = Field(ge=0)
    kv_cache_utilization_pct: float = Field(ge=0, le=100)
    preemption_rate_pct: float = Field(ge=0, le=100)
    sla_adherence_pct: float = Field(ge=0, le=100, description="Percentage meeting SLA (<1.5s TTFT, <3.5s total)")


class ServingBenchmarkReport(BaseModel):
    """Comprehensive self-hosted vLLM inference and concurrency simulation report."""
    model_id: str
    precision: QuantizationPrecision
    hardware_name: str
    tensor_parallel_size: int
    sizing: MemorySizingResult
    concurrency_ramp: List[ConcurrencyStepResult] = Field(default_factory=list)
    max_recommended_concurrency: int = Field(description="Peak concurrency before SLA drops below 95%")
    peak_throughput_tokens_sec: float
    notes: Optional[str] = None


class CostCrossoverComparison(BaseModel):
    """Cost comparison between self-hosted dedicated GPUs and hosted cloud APIs."""
    model_id: str
    hardware_name: str
    tensor_parallel_size: int
    gpu_hourly_cost_usd: Decimal
    monthly_hosting_cost_usd: Decimal = Field(description="GPU cost * 730 hrs + 15% network/storage buffer")
    cloud_api_input_per_1m: Decimal
    cloud_api_output_per_1m: Decimal
    avg_prompt_tokens: int = Field(default=1200)
    avg_completion_tokens: int = Field(default=350)
    cost_per_query_cloud_usd: Decimal
    break_even_monthly_queries: int = Field(description="Monthly query volume where self-hosting becomes cheaper")
    break_even_daily_queries: int = Field(description="Daily query volume equivalent")
    is_economically_viable_for_msu: bool = Field(description="True if MSU's estimated admissions traffic exceeds break-even")
    recommendation_summary: str
