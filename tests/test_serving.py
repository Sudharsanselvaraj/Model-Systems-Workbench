"""Unit tests for Phase 6 vLLM serving simulation, hardware sizing, and cost crossover."""

from decimal import Decimal
from pathlib import Path
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.serving import (
    CostCrossoverAnalyzer,
    HARDWARE_PROFILES,
    HardwareSizingEngine,
    MODEL_ARCHITECTURES,
    QuantizationPrecision,
    ServingBenchmarkSimulator,
    ServingReportGenerator,
    get_hardware_profile,
    get_model_architecture,
)

runner = CliRunner()


def test_hardware_profiles_and_architectures():
    """Verify standard accelerator profiles and transformer architectures are correctly loaded."""
    assert "a100_80gb" in HARDWARE_PROFILES
    assert "rtx_4090" in HARDWARE_PROFILES
    assert "mac_studio_128gb" in HARDWARE_PROFILES

    a100 = get_hardware_profile("a100_80gb")
    assert a100.vram_gb == 80.0
    assert a100.memory_bandwidth_gb_s > 2000.0

    qwen72b = get_model_architecture("qwen/qwen-2.5-72b")
    assert qwen72b.param_count_billions > 70.0
    assert qwen72b.is_gqa is True  # 8 KV heads vs 64 attention heads

    qwen7b = get_model_architecture("qwen/qwen-2.5-7b")
    assert qwen7b.is_gqa is True


def test_hardware_sizing_engine_calculations():
    """Verify memory allocation math, GQA KV cache sizing, and OOM detection."""
    qwen72b = MODEL_ARCHITECTURES["qwen/qwen-2.5-72b"]
    a100 = HARDWARE_PROFILES["a100_80gb"]
    rtx4090 = HARDWARE_PROFILES["rtx_4090"]

    # 1. 72B in BF16 on 1x RTX 4090 (24GB) must fail with OOM
    res_oom = HardwareSizingEngine.calculate_sizing(
        model=qwen72b,
        gpu=rtx4090,
        precision=QuantizationPrecision.BF16,
        tensor_parallel_size=1,
    )
    assert res_oom.is_deployable is False
    assert "VRAM Exceeded" in res_oom.limiting_factor

    # 2. 72B in BF16 on 2x A100 (160GB total VRAM) must be deployable
    res_deploy = HardwareSizingEngine.calculate_sizing(
        model=qwen72b,
        gpu=a100,
        precision=QuantizationPrecision.BF16,
        tensor_parallel_size=2,
    )
    assert res_deploy.is_deployable is True
    assert res_deploy.weights_memory_gb > 130.0
    assert res_deploy.kv_cache_allocated_gb > 10.0
    assert res_deploy.max_concurrent_sessions_8k >= 4

    # 3. 72B in AWQ_INT4 on 1x Mac Studio (128GB)
    mac = HARDWARE_PROFILES["mac_studio_128gb"]
    res_awq = HardwareSizingEngine.calculate_sizing(
        model=qwen72b,
        gpu=mac,
        precision=QuantizationPrecision.AWQ_INT4,
        tensor_parallel_size=1,
    )
    assert res_awq.is_deployable is True
    # 4-bit weights should be around ~40 GB
    assert 35.0 <= res_awq.weights_memory_gb <= 45.0
    assert res_awq.max_concurrent_sessions_8k > 20

    # 4. recommend_min_tp check
    min_tp_a100 = HardwareSizingEngine.recommend_min_tp(qwen72b, a100, QuantizationPrecision.BF16)
    assert min_tp_a100 == 2


def test_serving_benchmark_simulator_concurrency_ramp():
    """Verify simulation calculates TTFT, TPOT, and throughput scaling across concurrency levels."""
    model = MODEL_ARCHITECTURES["qwen/qwen-2.5-14b"]
    gpu = HARDWARE_PROFILES["a100_80gb"]

    sim = ServingBenchmarkSimulator(avg_prompt_tokens=1000, avg_completion_tokens=250)
    report = sim.simulate_ramp(
        model=model,
        gpu=gpu,
        precision=QuantizationPrecision.BF16,
        tensor_parallel_size=1,
        concurrency_levels=[1, 5, 10, 25],
    )

    assert report.model_id == model.model_id
    assert len(report.concurrency_ramp) == 4
    assert report.peak_throughput_tokens_sec > 0.0

    step1 = report.concurrency_ramp[0]
    step25 = report.concurrency_ramp[3]

    # Concurrency 25 throughput must be higher than concurrency 1 (batch amortization)
    assert step25.output_tokens_per_second > step1.output_tokens_per_second
    # Concurrency 25 P95 TTFT must be higher than concurrency 1 (queueing / interference)
    assert step25.p95_ttft_ms >= step1.p95_ttft_ms


def test_cost_crossover_analyzer():
    """Verify break-even inquiry volume calculations against cloud API pricing."""
    model = MODEL_ARCHITECTURES["qwen/qwen-2.5-72b"]
    a100 = HARDWARE_PROFILES["a100_80gb"]
    mac = HARDWARE_PROFILES["mac_studio_128gb"]

    analyzer = CostCrossoverAnalyzer()

    # 1. 2x A100 cluster ($3.28/hr) vs Gemini Flash ($0.075 / $0.30)
    comp = analyzer.analyze(
        model=model,
        gpu=a100,
        cloud_api_input_per_1m=Decimal("0.0750"),
        cloud_api_output_per_1m=Decimal("0.3000"),
        tensor_parallel_size=2,
        estimated_monthly_msu_queries=60000,
    )
    assert comp.monthly_hosting_cost_usd > Decimal("2000.00")
    assert comp.cost_per_query_cloud_usd < Decimal("0.001")
    assert comp.break_even_monthly_queries > 5_000_000
    assert comp.is_economically_viable_for_msu is False
    assert "Cloud API is significantly more economical" in comp.recommendation_summary

    # 2. On-premise hardware ($0.00/hr)
    comp_mac = analyzer.analyze(
        model=model,
        gpu=mac,
        cloud_api_input_per_1m=Decimal("0.0750"),
        cloud_api_output_per_1m=Decimal("0.3000"),
        tensor_parallel_size=1,
    )
    assert comp_mac.monthly_hosting_cost_usd == Decimal("0.00")
    assert comp_mac.break_even_monthly_queries == 0
    assert comp_mac.is_economically_viable_for_msu is True


def test_serving_report_generator(tmp_path: Path):
    """Verify Markdown generation and report file persistence."""
    model = MODEL_ARCHITECTURES["qwen/qwen-2.5-7b"]
    gpu = HARDWARE_PROFILES["rtx_4090"]

    sim = ServingBenchmarkSimulator()
    b_report = sim.simulate_ramp(model, gpu, tensor_parallel_size=1, concurrency_levels=[1, 5])
    cost_comp = CostCrossoverAnalyzer().analyze(
        model, gpu, Decimal("0.10"), Decimal("0.40"), tensor_parallel_size=1
    )

    md = ServingReportGenerator.generate_markdown(b_report, cost_comp)
    assert "Hardware VRAM Allocation Breakdown" in md
    assert "Progressive Concurrency Ramp" in md
    assert "Cost-Crossover" in md
    assert "qwen/qwen-2.5-7b" in md

    report_path = ServingReportGenerator.save_report(b_report, cost_comp, output_dir=tmp_path)
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8") == md


def test_cli_simulate_serving_command(tmp_path: Path):
    """Verify simulate-serving CLI command executes successfully and outputs rich tables."""
    result = runner.invoke(
        app,
        [
            "simulate-serving",
            "--model",
            "qwen/qwen-2.5-7b",
            "--hardware",
            "rtx_4090",
            "--tp",
            "1",
            "--precision",
            "BF16",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "Hardware Memory Sizing & KV-Cache Allocation" in result.output
    assert "Progressive Concurrency Ramp Simulation" in result.output
    assert "Cost Crossover Analysis" in result.output
    assert "Serving benchmark report saved to" in result.output
