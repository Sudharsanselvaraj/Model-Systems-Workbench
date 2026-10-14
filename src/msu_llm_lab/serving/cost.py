"""Cost crossover analysis engine comparing dedicated self-hosted GPUs with hosted cloud APIs."""

from decimal import Decimal
from typing import Optional
from msu_llm_lab.serving.models import CostCrossoverComparison, GPUProfile, ModelArchitecture
from msu_llm_lab.serving.profiles import HARDWARE_PROFILES, MODEL_ARCHITECTURES, get_hardware_profile, get_model_architecture


class CostCrossoverAnalyzer:
    """Calculates break-even inquiry volume thresholds between self-hosting and cloud APIs."""

    def __init__(
        self,
        avg_prompt_tokens: int = 1200,
        avg_completion_tokens: int = 350,
        hours_per_month: float = 730.0,
        overhead_multiplier: float = 1.15,  # 15% buffer for storage, egress, and host VM
    ):
        self.avg_prompt_tokens = avg_prompt_tokens
        self.avg_completion_tokens = avg_completion_tokens
        self.hours_per_month = Decimal(str(hours_per_month))
        self.overhead_multiplier = Decimal(str(overhead_multiplier))

    def analyze(
        self,
        model: ModelArchitecture,
        gpu: GPUProfile,
        cloud_api_input_per_1m: Decimal,
        cloud_api_output_per_1m: Decimal,
        tensor_parallel_size: int = 1,
        estimated_monthly_msu_queries: int = 60000,  # ~2,000 inquiries/day during peak intake
    ) -> CostCrossoverComparison:
        """Computes exact monthly and daily crossover points where self-hosting becomes cheaper."""
        # 1. Cloud API Cost Per Single Query
        prompt_cost = (Decimal(str(self.avg_prompt_tokens)) / Decimal("1000000")) * cloud_api_input_per_1m
        comp_cost = (Decimal(str(self.avg_completion_tokens)) / Decimal("1000000")) * cloud_api_output_per_1m
        cost_per_query_cloud = prompt_cost + comp_cost

        # 2. Self-Hosted Monthly Cost
        hourly_gpu_total = gpu.hourly_rate_usd * Decimal(str(tensor_parallel_size))
        monthly_base = hourly_gpu_total * self.hours_per_month
        monthly_hosting_cost = monthly_base * self.overhead_multiplier

        # 3. Break-even Query Threshold
        if cost_per_query_cloud > Decimal("0.0000001"):
            break_even_monthly = int(monthly_hosting_cost / cost_per_query_cloud)
            break_even_daily = int(break_even_monthly / Decimal("30.42"))
        else:
            break_even_monthly = 999_999_999
            break_even_daily = 999_999_999

        # If on-premise owned hardware (e.g. Mac Studio M2 Ultra @ $0/hr cloud rate)
        if gpu.hourly_rate_usd == Decimal("0.00"):
            break_even_monthly = 0
            break_even_daily = 0
            is_viable = True
            rec = (
                f"On-premise hardware deployment ({gpu.name}) has zero recurring GPU rental cost. "
                f"Economically superior to cloud API at all query volumes."
            )
        else:
            is_viable = estimated_monthly_msu_queries >= break_even_monthly
            if is_viable:
                rec = (
                    f"Self-hosting on {tensor_parallel_size}x {gpu.name} is economically justified "
                    f"because MSU monthly inquiries ({estimated_monthly_msu_queries:,}) exceed break-even ({break_even_monthly:,})."
                )
            else:
                ratio = (break_even_monthly / max(1, estimated_monthly_msu_queries))
                rec = (
                    f"Cloud API is significantly more economical. Break-even requires {break_even_monthly:,} queries/mo "
                    f"({ratio:.1f}x MSU estimated intake volume of {estimated_monthly_msu_queries:,}/mo). "
                    f"Self-hosting dedicated GPUs is only justified if strict data residency or on-prem airgap is mandated."
                )

        return CostCrossoverComparison(
            model_id=model.model_id,
            hardware_name=gpu.name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_hourly_cost_usd=hourly_gpu_total,
            monthly_hosting_cost_usd=monthly_hosting_cost.quantize(Decimal("0.01")),
            cloud_api_input_per_1m=cloud_api_input_per_1m,
            cloud_api_output_per_1m=cloud_api_output_per_1m,
            avg_prompt_tokens=self.avg_prompt_tokens,
            avg_completion_tokens=self.avg_completion_tokens,
            cost_per_query_cloud_usd=cost_per_query_cloud.quantize(Decimal("0.000001")),
            break_even_monthly_queries=break_even_monthly,
            break_even_daily_queries=break_even_daily,
            is_economically_viable_for_msu=is_viable,
            recommendation_summary=rec,
        )
