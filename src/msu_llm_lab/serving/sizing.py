"""Hardware memory sizing engine for vLLM transformer deployments."""

from typing import List, Optional
from msu_llm_lab.serving.models import (
    GPUProfile,
    MemorySizingResult,
    ModelArchitecture,
    QuantizationPrecision,
)
from msu_llm_lab.serving.profiles import HARDWARE_PROFILES, MODEL_ARCHITECTURES, get_hardware_profile, get_model_architecture


class HardwareSizingEngine:
    """Calculates exact GPU memory allocations and maximum concurrent sessions for vLLM."""

    @staticmethod
    def get_bytes_per_param(precision: QuantizationPrecision) -> float:
        """Returns bytes required per parameter based on quantization precision."""
        if precision in (QuantizationPrecision.FP16, QuantizationPrecision.BF16):
            return 2.0
        elif precision == QuantizationPrecision.INT8:
            return 1.0
        elif precision in (QuantizationPrecision.AWQ_INT4, QuantizationPrecision.GPTQ_INT4):
            # 4-bit weights + scale/zero overhead
            return 0.55
        return 2.0

    @classmethod
    def calculate_sizing(
        cls,
        model: ModelArchitecture,
        gpu: GPUProfile,
        precision: QuantizationPrecision = QuantizationPrecision.BF16,
        tensor_parallel_size: int = 1,
        kv_cache_dtype_bytes: float = 2.0,  # 2 bytes for FP16/BF16 KV-cache
    ) -> MemorySizingResult:
        """Computes complete memory allocation, KV-cache capacity, and concurrency bounds."""
        bytes_per_param = cls.get_bytes_per_param(precision)
        
        # 1. Model Weights Memory with 5% framework padding
        weights_bytes = model.param_count_billions * 1e9 * bytes_per_param * 1.05
        weights_memory_gb = round(weights_bytes / (1024 ** 3), 2)

        # 2. Total GPU Cluster VRAM
        total_gpu_vram_gb = round(gpu.vram_gb * tensor_parallel_size, 2)

        # 3. Static CUDA context and PyTorch runtime overhead per GPU
        cuda_runtime_overhead_gb = round(1.2 * tensor_parallel_size, 2)

        # 4. Activation working memory buffer
        # In vLLM PagedAttention, activation overhead scales mildly with TP
        activation_overhead_gb = round(1.5 + (0.3 * tensor_parallel_size), 2)

        # 5. KV-Cache Per Token calculation (Transformer Attention with GQA support)
        # Head dimension = hidden_size / num_attention_heads
        head_dim = model.hidden_size / model.num_attention_heads
        # KV Cache per token = 2 (Key + Value) * num_layers * num_kv_heads * head_dim * bytes_per_element
        kv_cache_per_token_bytes = 2 * model.num_layers * model.num_kv_heads * head_dim * kv_cache_dtype_bytes

        # 6. Available VRAM remaining for PagedAttention KV-Cache blocks
        static_usage_gb = weights_memory_gb + activation_overhead_gb + cuda_runtime_overhead_gb
        available_kv_gb = max(0.0, total_gpu_vram_gb - static_usage_gb)

        is_deployable = (total_gpu_vram_gb >= static_usage_gb) and (available_kv_gb >= 2.0)

        # 7. Concurrency bounds at standard context sizes (8k and 32k)
        gb_in_bytes = 1024 ** 3
        memory_per_8k_session_gb = (8192 * kv_cache_per_token_bytes) / gb_in_bytes
        memory_per_32k_session_gb = (32768 * kv_cache_per_token_bytes) / gb_in_bytes

        if is_deployable and memory_per_8k_session_gb > 0:
            max_sessions_8k = int(available_kv_gb / memory_per_8k_session_gb)
            max_sessions_32k = int(available_kv_gb / memory_per_32k_session_gb)
        else:
            max_sessions_8k = 0
            max_sessions_32k = 0

        # 8. Limiting factor diagnostic
        if not is_deployable:
            # Estimate minimum TP required directly without recursion
            min_tp_est = max(1, int((static_usage_gb + 4.0) / max(1.0, gpu.vram_gb - 2.0)) + 1)
            limiting_factor = (
                f"VRAM Exceeded (Required {static_usage_gb:.1f} GB, Total {total_gpu_vram_gb:.1f} GB). "
                f"Estimated minimum Tensor Parallelism required: TP >= {min_tp_est}"
            )
        elif available_kv_gb < 8.0:
            limiting_factor = "KV-Cache Constrained: Insufficient buffer for high concurrency ramp."
        elif max_sessions_8k >= 100:
            limiting_factor = "Compute / Bandwidth Bound (High VRAM capacity for 100+ concurrent users)"
        else:
            limiting_factor = f"Moderate KV-Cache Pool ({available_kv_gb:.1f} GB available)"

        return MemorySizingResult(
            model_id=model.model_id,
            precision=precision,
            tensor_parallel_size=tensor_parallel_size,
            gpu_profile=gpu,
            total_gpu_vram_gb=total_gpu_vram_gb,
            weights_memory_gb=weights_memory_gb,
            kv_cache_per_token_bytes=round(kv_cache_per_token_bytes, 1),
            kv_cache_allocated_gb=round(available_kv_gb, 2),
            activation_overhead_gb=activation_overhead_gb,
            cuda_runtime_overhead_gb=cuda_runtime_overhead_gb,
            total_memory_required_gb=round(static_usage_gb, 2),
            is_deployable=is_deployable,
            max_concurrent_sessions_8k=max_sessions_8k,
            max_concurrent_sessions_32k=max_sessions_32k,
            limiting_factor=limiting_factor,
        )

    @classmethod
    def recommend_min_tp(
        cls,
        model: ModelArchitecture,
        gpu: GPUProfile,
        precision: QuantizationPrecision = QuantizationPrecision.BF16,
    ) -> int:
        """Calculates the minimum Tensor Parallelism required to deploy without OOM."""
        bytes_per_param = cls.get_bytes_per_param(precision)
        weights_bytes = model.param_count_billions * 1e9 * bytes_per_param * 1.05
        weights_gb = weights_bytes / (1024 ** 3)

        for tp in [1, 2, 4, 8]:
            total_vram = gpu.vram_gb * tp
            cuda_overhead = 1.2 * tp
            activation_overhead = 1.5 + (0.3 * tp)
            static_gb = weights_gb + cuda_overhead + activation_overhead
            available_kv = total_vram - static_gb
            if available_kv >= 4.0:
                return tp
        return 8
