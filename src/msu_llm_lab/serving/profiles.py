"""Predefined hardware accelerator profiles and transformer model architecture specifications."""

from decimal import Decimal
from typing import Dict, List, Optional
from msu_llm_lab.serving.models import GPUProfile, ModelArchitecture, QuantizationPrecision


# ---------------------------------------------------------------------------
# Standard Hardware Accelerators
# ---------------------------------------------------------------------------

HARDWARE_PROFILES: Dict[str, GPUProfile] = {
    "rtx_4090": GPUProfile(
        name="NVIDIA GeForce RTX 4090 (24GB)",
        vram_gb=24.0,
        memory_bandwidth_gb_s=1008.0,
        fp16_tflops=165.0,
        hourly_rate_usd=Decimal("0.44"),
        interconnect="PCIe 4.0",
    ),
    "a100_80gb": GPUProfile(
        name="NVIDIA A100-SXM4 (80GB)",
        vram_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        fp16_tflops=312.0,
        hourly_rate_usd=Decimal("1.64"),
        interconnect="NVLink (600 GB/s)",
    ),
    "h100_80gb": GPUProfile(
        name="NVIDIA H100-SXM5 (80GB)",
        vram_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        fp16_tflops=989.0,
        hourly_rate_usd=Decimal("2.49"),
        interconnect="NVLink 4 (900 GB/s)",
    ),
    "l40s_48gb": GPUProfile(
        name="NVIDIA L40S (48GB)",
        vram_gb=48.0,
        memory_bandwidth_gb_s=864.0,
        fp16_tflops=366.0,
        hourly_rate_usd=Decimal("0.99"),
        interconnect="PCIe 4.0",
    ),
    "mac_studio_128gb": GPUProfile(
        name="Apple Mac Studio M2 Ultra (128GB Unified)",
        vram_gb=128.0,
        memory_bandwidth_gb_s=800.0,
        fp16_tflops=67.0,
        hourly_rate_usd=Decimal("0.00"),  # On-premise owned hardware
        interconnect="Unified Memory Architecture",
    ),
}


# ---------------------------------------------------------------------------
# Target Open-Weight Model Architectures
# ---------------------------------------------------------------------------

MODEL_ARCHITECTURES: Dict[str, ModelArchitecture] = {
    "qwen/qwen-2.5-7b": ModelArchitecture(
        model_id="qwen/qwen-2.5-7b",
        display_name="Qwen 2.5 7B Instruct",
        param_count_billions=7.61,
        hidden_size=3584,
        num_layers=28,
        num_attention_heads=28,
        num_kv_heads=4,  # Grouped Query Attention (GQA 7:1 ratio)
        vocab_size=152064,
        default_context_window=32768,
        supported_precisions=[QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4, QuantizationPrecision.INT8],
    ),
    "qwen/qwen-2.5-14b": ModelArchitecture(
        model_id="qwen/qwen-2.5-14b",
        display_name="Qwen 2.5 14B Instruct",
        param_count_billions=14.77,
        hidden_size=5120,
        num_layers=48,
        num_attention_heads=40,
        num_kv_heads=8,  # GQA 5:1 ratio
        vocab_size=152064,
        default_context_window=32768,
        supported_precisions=[QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4, QuantizationPrecision.INT8],
    ),
    "qwen/qwen-2.5-32b": ModelArchitecture(
        model_id="qwen/qwen-2.5-32b",
        display_name="Qwen 2.5 32B Instruct",
        param_count_billions=32.5,
        hidden_size=5120,
        num_layers=64,
        num_attention_heads=40,
        num_kv_heads=8,  # GQA 5:1 ratio
        vocab_size=152064,
        default_context_window=32768,
        supported_precisions=[QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4, QuantizationPrecision.INT8],
    ),
    "qwen/qwen-2.5-72b": ModelArchitecture(
        model_id="qwen/qwen-2.5-72b",
        display_name="Qwen 2.5 72B Instruct",
        param_count_billions=72.7,
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,  # GQA 8:1 ratio
        vocab_size=152064,
        default_context_window=32768,
        supported_precisions=[QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4, QuantizationPrecision.INT8],
    ),
    "meta/llama-3.3-70b": ModelArchitecture(
        model_id="meta/llama-3.3-70b",
        display_name="Llama 3.3 70B Instruct",
        param_count_billions=70.6,
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,  # GQA 8:1 ratio
        vocab_size=128256,
        default_context_window=131072,
        supported_precisions=[QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4, QuantizationPrecision.INT8],
    ),
    "meta/llama-3.1-8b": ModelArchitecture(
        model_id="meta/llama-3.1-8b",
        display_name="Llama 3.1 8B Instruct",
        param_count_billions=8.03,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,  # GQA 4:1 ratio
        vocab_size=128256,
        default_context_window=131072,
        supported_precisions=[QuantizationPrecision.BF16, QuantizationPrecision.AWQ_INT4, QuantizationPrecision.INT8],
    ),
}


def get_hardware_profile(key: str) -> GPUProfile:
    """Returns matching GPU profile or default A100."""
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    for k, profile in HARDWARE_PROFILES.items():
        if k in normalized or normalized in k:
            return profile
    return HARDWARE_PROFILES["a100_80gb"]


def get_model_architecture(model_id: str) -> ModelArchitecture:
    """Resolves model ID to architecture specification."""
    normalized = model_id.lower()
    for k, arch in MODEL_ARCHITECTURES.items():
        if k in normalized or arch.display_name.lower() in normalized:
            return arch
    # Default to Qwen 2.5 7B if unknown
    return MODEL_ARCHITECTURES["qwen/qwen-2.5-7b"]
