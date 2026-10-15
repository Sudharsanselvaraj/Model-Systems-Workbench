"""Sourced verification registry for benchmark participants.

Every fact carries a status, source URLs and the date it was checked. Nothing is marked verified
unless a primary source stated it; anything else is SECONDARY_ONLY, CONFLICTING or UNKNOWN.
This registry is separate from the pricing catalog and never changes what the budget layer charges.
"""

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

CHECKED_ON = date(2026, 10, 15)

OR_JEV_PROVIDERS = "https://openrouter.ai/typesafe/jev-1.14/providers"
OR_JEV_GUIDE = "https://openrouter.ai/docs/guides/community/jev"
OR_JEV_BLOG = "https://openrouter.ai/blog/insights/what-is-jev/"
TS_CHOICE = "https://docs.typesafe.ai/primitives/choice"
TS_PRIMITIVES = "https://docs.typesafe.ai/primitives"
OR_GPT41N = "https://openrouter.ai/openai/gpt-4.1-nano"
OR_ERRORS = "https://openrouter.ai/docs/api-reference/errors"
OR_STRUCT = "https://openrouter.ai/docs/features/structured-outputs"
OR_PRIVACY = "https://openrouter.ai/docs/features/privacy-and-logging"
OR_ENDPOINTS = "https://openrouter.ai/api/v1/models/{}/endpoints"
OR_CHAT_RESPONSE = "https://openrouter.ai/docs/api-reference/overview"
OR_PROVIDERS_API = "https://openrouter.ai/api/v1/providers"
OR_GPT41N_PROVIDERS = "https://openrouter.ai/openai/gpt-4.1-nano/providers"
OR_PROVIDER_ROUTING = "https://openrouter.ai/docs/features/provider-routing"
AGGREGATORS = "https://www.getmaxim.ai/bifrost/llm-cost-calculator/provider/openrouter/model/gpt-4.1-nano"


class FactStatus(str, Enum):
    VERIFIED_PRIMARY = "VERIFIED_PRIMARY"  # stated on the provider's or OpenRouter's own documentation
    USER_SUPPLIED = "USER_SUPPLIED"        # figure supplied by the project owner; must be re-verified before any live run
    SECONDARY_ONLY = "SECONDARY_ONLY"      # only third-party or summarized sources; not confirmed on a primary page
    CONFLICTING = "CONFLICTING"            # sources disagree
    UNKNOWN = "UNKNOWN"


class Protocol(str, Enum):
    CHAT_COMPLETIONS = "CHAT_COMPLETIONS"
    DECISIONS_API = "DECISIONS_API"


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Optional[Any] = None
    status: FactStatus = FactStatus.UNKNOWN
    source_urls: List[str] = Field(default_factory=list)
    checked_on: Optional[date] = None
    note: str = ""

    @model_validator(mode="after")
    def _honest(self):
        if self.status == FactStatus.UNKNOWN and self.value is not None:
            raise ValueError("an UNKNOWN fact must not carry a value; put reference figures in the note")
        if self.status != FactStatus.UNKNOWN and self.checked_on is None:
            raise ValueError("any non-UNKNOWN fact needs checked_on")
        if self.status not in (FactStatus.UNKNOWN, FactStatus.USER_SUPPLIED) and not self.source_urls:
            raise ValueError("a fact that is not UNKNOWN or USER_SUPPLIED needs source_urls")
        return self


def unknown(note: str) -> Fact:
    return Fact(note=note)


def fact(value: Any, status: FactStatus, urls: List[str], note: str = "") -> Fact:
    return Fact(value=value, status=status, source_urls=urls, checked_on=CHECKED_ON, note=note)


class RouteOption(BaseModel):
    """One provider endpoint as listed by OpenRouter's public endpoints metadata (prices converted to USD per 1M tokens)."""
    model_config = ConfigDict(extra="forbid")

    provider: str
    tag: str
    input_per_m: str
    output_per_m: str
    context: int
    max_completion: Optional[int] = None
    params: List[str] = Field(default_factory=list)
    uptime_1d: Optional[float] = None
    note: str = ""

    def supports(self, param: str) -> bool:
        return param in self.params


class RouteStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class RouteRecord(BaseModel):
    """Provider-route assumptions for one model. A route is BLOCKED until the provider and its policy are verified."""
    model_config = ConfigDict(extra="forbid")

    status: RouteStatus = RouteStatus.BLOCKED
    provider_preferences: Optional[Dict[str, Any]] = None
    route_input_price_per_1m_usd: Fact = Field(default_factory=lambda: unknown("Route-specific input price not verified."))
    route_output_price_per_1m_usd: Fact = Field(default_factory=lambda: unknown("Route-specific output price not verified."))
    serving_provider_field: Fact = Field(default_factory=lambda: unknown("Response field naming the serving provider is not documented."))
    assumptions: List[str] = Field(default_factory=list)
    blocked_reasons: List[str] = Field(default_factory=list)


class ModelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    display_name: str = ""
    roles: List[str] = Field(default_factory=lambda: ["CANDIDATE"])
    tasks: List[str] = Field(default_factory=lambda: ["INTENT", "LEAD"])
    protocol: Protocol
    requested_model_id: Fact
    endpoint: Fact
    request_fields: Fact
    response_shape: Fact
    output_type: Fact
    resolved_version_reporting: Fact
    usage_fields: Fact
    input_price_per_1m_usd: Fact
    output_price_per_1m_usd: Fact
    context_limit_tokens: Fact
    provider_route: Fact
    structured_output_support: Fact
    data_policy: Fact
    error_codes: Fact
    route: RouteRecord = Field(default_factory=RouteRecord)
    candidate_routes: List[RouteOption] = Field(default_factory=list)
    pinned_tag: Optional[str] = None
    production_reported_name: Optional[str] = None
    production_parity: Fact = Field(default_factory=lambda: unknown("Not applicable or not established."))
    open_decisions: List[str] = Field(default_factory=list)

    def pinned_route(self) -> Optional[RouteOption]:
        return next((r for r in self.candidate_routes if r.tag == self.pinned_tag), None)

    def readiness_blockers(self) -> List[str]:
        """Everything a live run needs, including student-derived data: facts not VERIFIED_PRIMARY, plus the route."""
        required = ("requested_model_id", "endpoint", "request_fields", "response_shape", "usage_fields",
                    "input_price_per_1m_usd", "output_price_per_1m_usd", "data_policy")
        out = [n for n in required if getattr(self, n).status != FactStatus.VERIFIED_PRIMARY]
        return out + (["route"] if self.route.status != RouteStatus.READY else [])

    def synthetic_blockers(self, stage: str = "pilot") -> List[str]:
        """What must be resolved before a synthetic-only live call. The data policy is not needed for synthetic data.

        stage="pilot" (default, strict): every required fact must be VERIFIED_PRIMARY.
        stage="smoke": the response schema facts may still be SECONDARY_ONLY, because a smoke test exists to confirm them and
        the adapters parse strictly and fail safely on deviation. UNKNOWN or CONFLICTING facts still block.
        """
        required = ("requested_model_id", "endpoint", "request_fields", "response_shape", "usage_fields",
                    "input_price_per_1m_usd", "output_price_per_1m_usd")
        schema_facts = {"response_shape", "usage_fields"}
        out = []
        for n in required:
            st = getattr(self, n).status
            if st == FactStatus.VERIFIED_PRIMARY:
                continue
            if stage == "smoke" and n in schema_facts and st == FactStatus.SECONDARY_ONLY:
                continue
            out.append(n)
        return out + (["route"] if self.route.status != RouteStatus.READY else [])

    def student_data_blockers(self) -> List[str]:
        """Unknown policy means no student-derived data may be sent."""
        out = self.synthetic_blockers()
        return out + (["data_policy"] if self.data_policy.status != FactStatus.VERIFIED_PRIMARY and "data_policy" not in out else [])


JEV_1_14 = ModelRecord(
    key="jev-1.14",
    display_name="Jev 1.14 (TypeSafe)",
    protocol=Protocol.DECISIONS_API,
    requested_model_id=fact("typesafe/jev-1.14", FactStatus.VERIFIED_PRIMARY, [OR_JEV_GUIDE, OR_ENDPOINTS.format("typesafe/jev-1.14")],
        "OpenRouter's endpoints metadata lists the model (name 'TypeSafe: Jev 1.14', one endpoint named typesafe/jev-1.14-20261014, modality text+image->decisions, "
        "no supported_parameters, 262,144 context, max completion 235,929). Guide names it as the current model to pin; alias ~typesafe/jev-latest tracks the newest release; 1.13 still available. "
        "Other pages still describe 1.13 as current, so availability to this account is confirmed only by a real call."),
    endpoint=fact("POST https://openrouter.ai/api/alpha/decisions", FactStatus.VERIFIED_PRIMARY, [OR_JEV_GUIDE, OR_JEV_BLOG],
        "Alpha path: schema may change. Not a Chat Completions endpoint."),
    request_fields=fact(
        {"top_level": ["model", "state", "questions"],
         "choice": {"type": "choice", "instructions": "required", "criteria": "required map option->description, max 255 options"},
         "noul": {"type": "noul", "instructions": "required", "criteria": "optional"}},
        FactStatus.VERIFIED_PRIMARY, [TS_CHOICE, TS_PRIMITIVES, OR_JEV_GUIDE],
        "Choice and Noul field names come from TypeSafe's own docs; the OpenRouter guide documents only the top-level fields."),
    response_shape=fact(
        {"envelope": ["model", "answers", "usage", "id", "provider"],
         "choice": ["type", "choice", "probabilities", "confidence"], "noul": ["type", "noul"]},
        FactStatus.SECONDARY_ONLY, [OR_JEV_BLOG, TS_CHOICE, TS_PRIMITIVES],
        "Choice/Noul answer keys are in TypeSafe's docs; the OpenRouter envelope (answers, id, provider) comes from an example for jev-1.13 only."),
    output_type=fact("typed structured answer; no free text or reasoning", FactStatus.VERIFIED_PRIMARY, [OR_JEV_GUIDE, TS_PRIMITIVES],
        "Choice: selected option + per-option probabilities + separate confidence (derived from probability spread; not a stated accuracy). "
        "Noul: probability of yes only; no confidence field."),
    resolved_version_reporting=fact("response 'model' field holds the served snapshot, e.g. typesafe/jev-1.13-20260917",
        FactStatus.SECONDARY_ONLY, [OR_JEV_BLOG], "Seen for 1.13 only; confirm for 1.14 on the first real response."),
    usage_fields=fact({"input_tokens": "int", "output_tokens": "int", "cost": "USD"}, FactStatus.SECONDARY_ONLY, [OR_JEV_BLOG, OR_JEV_GUIDE],
        "Guide confirms usage.cost; token field names (input_tokens/output_tokens) come from the blog example and TypeSafe's direct-API docs."),
    input_price_per_1m_usd=fact("0.042", FactStatus.VERIFIED_PRIMARY, [OR_JEV_PROVIDERS],
        "OpenRouter's providers page for typesafe/jev-1.14 lists $0.042 per 1M input tokens (one listed provider: TypeSafe). "
        "Matches the owner's own check on 2026-10-15. Re-verify before any live run."),
    output_price_per_1m_usd=fact("0", FactStatus.VERIFIED_PRIMARY, [OR_JEV_PROVIDERS, OR_JEV_GUIDE],
        "OpenRouter's providers page lists $0 per 1M output tokens; the guide states output tokens are free."),
    context_limit_tokens=fact("262,144 (page: 262K)", FactStatus.VERIFIED_PRIMARY, [OR_JEV_PROVIDERS, OR_JEV_GUIDE],
        "Both OpenRouter pages state 262K; the guide adds that requests over ~32,000 bytes of text run in a slower, less accurate "
        "long-context mode. Older third-party figures (32K/64K) were superseded."),
    provider_route=fact("TypeSafe (the only provider listed)", FactStatus.VERIFIED_PRIMARY, [OR_JEV_PROVIDERS],
        "Released Oct 14, 2026 per the page. No other providers or routes are listed."),
    candidate_routes=[RouteOption(provider="TypeSafe", tag="typesafe", input_per_m="0.042", output_per_m="0", context=262144,
                                  max_completion=235929, params=[], uptime_1d=100.0, note="only endpoint; dated snapshot typesafe/jev-1.14-20261014")],
    pinned_tag="typesafe",
    route=RouteRecord(
        status=RouteStatus.READY,
        route_input_price_per_1m_usd=fact("0.042", FactStatus.VERIFIED_PRIMARY, [OR_JEV_PROVIDERS], "Provider-level listing for TypeSafe."),
        route_output_price_per_1m_usd=fact("0", FactStatus.VERIFIED_PRIMARY, [OR_JEV_PROVIDERS], "Provider-level listing for TypeSafe."),
        serving_provider_field=fact("provider", FactStatus.SECONDARY_ONLY, [OR_JEV_BLOG], "Seen in a jev-1.13 example response."),
        assumptions=["Served by TypeSafe through OpenRouter's Decisions API; provider routing preferences are not documented for it, so none are sent.",
                     "READY covers synthetic-only use. The TypeSafe data policy is UNKNOWN, so student-derived data stays blocked."],
        blocked_reasons=[],
    ),
    structured_output_support=fact("native: typed answers by design", FactStatus.VERIFIED_PRIMARY, [OR_JEV_GUIDE], ""),
    data_policy=unknown("Retention and training policy not found for Jev; OpenRouter says policies are per provider."),
    error_codes=unknown("The Jev guide documents no error codes. OpenRouter's general codes may apply but are unconfirmed for this endpoint."),
    open_decisions=[
        "Noul gives only P(yes); no threshold is documented. An evaluation threshold must be chosen and recorded by the project, never defaulted.",
        "Jev confidence is derived from probability spread and is not documented as calibrated accuracy; report it separately from correctness.",
        "Token field names and the OpenRouter envelope must be captured from one approved synthetic call before the adapter is trusted.",
    ],
)

def _ep(provider: str, tag: str, inp: str, out: str, ctx: int, maxc: Optional[int], params: str, up: Optional[float], note: str = "") -> RouteOption:
    return RouteOption(provider=provider, tag=tag, input_per_m=inp, output_per_m=out, context=ctx, max_completion=maxc,
                       params=params.split(), uptime_1d=up, note=note)


_STD = "temperature seed response_format structured_outputs"
_CHAT_FIELDS = ["model", "messages", "temperature", "max_tokens", "seed", "response_format", "stop", "tools", "tool_choice", "stream", "provider"]
_POLICY_NOTE = ("Provider privacy-policy URLs exist in OpenRouter's providers metadata, but no retention or training finding was retrieved, "
                "so the policy is UNKNOWN and student-derived data stays blocked.")


def _chat_record(key: str, display: str, roles: List[str], tasks: List[str], model_id: str, endpoints: List[RouteOption], pin_tag: str,
                 *, production_name: Optional[str] = None, created: str = "", extra_decisions: Optional[List[str]] = None,
                 extra_assumptions: Optional[List[str]] = None, version_note: str = "") -> ModelRecord:
    """A chat-completions participant built from OpenRouter's public endpoint metadata (checked 2026-10-15)."""
    src = [OR_ENDPOINTS.format(model_id)]
    pin = next(e for e in endpoints if e.tag == pin_tag)
    prefs = {"order": [pin.tag], "allow_fallbacks": False}
    temp = pin.supports("temperature")
    return ModelRecord(
        key=key, display_name=display, roles=roles, tasks=tasks, protocol=Protocol.CHAT_COMPLETIONS,
        requested_model_id=fact(model_id, FactStatus.VERIFIED_PRIMARY, src, f"Listed by OpenRouter's endpoints metadata{created}."),
        endpoint=fact("POST https://openrouter.ai/api/v1/chat/completions", FactStatus.VERIFIED_PRIMARY, [f"https://openrouter.ai/{model_id}"], "Bearer authentication."),
        request_fields=fact(_CHAT_FIELDS, FactStatus.VERIFIED_PRIMARY, [f"https://openrouter.ai/{model_id}"], "Optional fields vary by model and endpoint; see route parameter support."),
        response_shape=fact("chat.completion: id, model, choices[].message.content, usage", FactStatus.VERIFIED_PRIMARY, [OR_CHAT_RESPONSE],
                            "A 200 body may carry an `error` object instead of choices. The docs do not list a `provider` field in the response."),
        output_type=fact("free text; JSON only if response_format is requested (not used in the initial comparison)", FactStatus.VERIFIED_PRIMARY, [OR_CHAT_RESPONSE], ""),
        resolved_version_reporting=fact("response `model` may differ from the requested model", FactStatus.VERIFIED_PRIMARY, [OR_CHAT_RESPONSE],
                                        "Documented as possibly different; it does not explicitly identify the serving provider. " + version_note),
        usage_fields=fact({"prompt_tokens": "int", "completion_tokens": "int", "total_tokens": "int", "cost": "credits (optional)"},
                          FactStatus.VERIFIED_PRIMARY, [OR_CHAT_RESPONSE], "usage is returned for non-streaming requests; `cost` is described as credits."),
        input_price_per_1m_usd=fact(pin.input_per_m, FactStatus.VERIFIED_PRIMARY, src, f"Price of the pinned route `{pin.tag}`."),
        output_price_per_1m_usd=fact(pin.output_per_m, FactStatus.VERIFIED_PRIMARY, src, f"Price of the pinned route `{pin.tag}`."),
        context_limit_tokens=fact(pin.context, FactStatus.VERIFIED_PRIMARY, src, f"Pinned route `{pin.tag}`; max completion {pin.max_completion}."),
        provider_route=fact(f"{pin.provider} (`{pin.tag}`)", FactStatus.VERIFIED_PRIMARY, src,
                            "PROPOSED PIN: the endpoint exists in OpenRouter's listing; the project owner must approve the pin."),
        structured_output_support=fact("response_format and structured_outputs listed" if pin.supports("structured_outputs") else "not listed on the pinned route",
                                       FactStatus.VERIFIED_PRIMARY, src, "Not used for the initial comparison."),
        data_policy=unknown(_POLICY_NOTE),
        error_codes=fact([400, 401, 402, 403, 404, 429, 502], FactStatus.VERIFIED_PRIMARY, [OR_ERRORS, f"https://openrouter.ai/{model_id}"],
                         "502 failed generations are not billed. 408, 500, 503, 504 appear in OpenRouter's general error docs; 524 and 529 are status constants only."),
        route=RouteRecord(
            status=RouteStatus.READY, provider_preferences=prefs,
            route_input_price_per_1m_usd=fact(pin.input_per_m, FactStatus.VERIFIED_PRIMARY, src, f"Endpoint `{pin.tag}`."),
            route_output_price_per_1m_usd=fact(pin.output_per_m, FactStatus.VERIFIED_PRIMARY, src, f"Endpoint `{pin.tag}`."),
            serving_provider_field=unknown("The chat-completions response docs do not list a `provider` field; capture it only if present. The pin plus allow_fallbacks=false fixes the route."),
            assumptions=[f"Proposed pin `{pin.tag}` with allow_fallbacks=false (no silent fallback). Owner approval of the route is part of the run plan.",
                         ("Temperature is supported on the pinned route." if temp else "Temperature is NOT listed on the pinned route, so requests omit it."),
                         "Plain-text output; no response_format.", *(extra_assumptions or [])],
            blocked_reasons=[],
        ),
        candidate_routes=endpoints, pinned_tag=pin_tag, production_reported_name=production_name,
        production_parity=(unknown("Production deployment unknown until the Dify export; this OpenRouter route is a benchmark route only.") if production_name
                           else unknown("Not a production-reported participant.")),
        open_decisions=["Owner approval of the pinned provider route is required.",
                        "Reasoning-capable models may spend max_tokens on reasoning; confirm the small token caps do not truncate outputs in the smoke test.",
                        *(extra_decisions or [])],
    )


_O = "https://openrouter.ai/"
_ENDPOINTS_NANO = [_ep("Azure", "azure", "0.10", "0.40", 1047576, 942818, "temperature top_p max_completion_tokens seed response_format structured_outputs tools tool_choice", 97.52),
                   _ep("OpenAI", "openai", "0.10", "0.40", 1047576, 32768, "temperature top_p max_tokens seed response_format structured_outputs tools tool_choice", 100.0),
                   _ep("Azure", "azure/swedencentral", "0.11", "0.44", 1047576, 942818, "temperature top_p max_completion_tokens seed response_format structured_outputs tools tool_choice", 100.0)]
_ENDPOINTS_4OMINI = [_ep("Azure", "azure", "0.15", "0.60", 128000, 16384, _STD + " max_completion_tokens", 99.79),
                     _ep("OpenAI", "openai", "0.15", "0.60", 128000, 16384, _STD + " max_tokens tools tool_choice", 100.0),
                     _ep("Azure", "azure/swedencentral", "0.165", "0.66", 128000, 16384, _STD + " max_completion_tokens", 100.0)]
_ENDPOINTS_41 = [_ep("Azure", "azure", "2.00", "8.00", 1047576, 942818, _STD + " max_completion_tokens", 94.44),
                 _ep("OpenAI", "openai", "2.00", "8.00", 1047576, 32768, _STD + " max_tokens", 100.0),
                 _ep("Azure", "azure/swedencentral", "2.20", "8.80", 1047576, 942818, _STD + " max_completion_tokens", 100.0)]
_ENDPOINTS_41MINI = [_ep("Azure", "azure", "0.40", "1.60", 1047576, 942818, _STD + " max_completion_tokens top_p", 98.82),
                     _ep("OpenAI", "openai", "0.40", "1.60", 1047576, 32768, _STD + " max_tokens top_p", 100.0),
                     _ep("Azure", "azure/swedencentral", "0.44", "1.76", 1047576, 942818, _STD + " max_completion_tokens top_p", 100.0)]
_ENDPOINTS_GEMINI = [_ep("Google", "google/global/flex", "0.15", "1.25", 1048576, 65536, "max_tokens seed response_format structured_outputs reasoning stop", 100.0, "flex tier"),
                     _ep("Google AI Studio", "google-ai-studio/flex", "0.15", "1.25", 1048576, 65536, "temperature top_p max_tokens seed response_format structured_outputs reasoning", 100.0, "flex tier"),
                     _ep("Google AI Studio", "google-ai-studio", "0.30", "2.50", 1048576, 65536, "temperature top_p max_tokens seed response_format structured_outputs reasoning", 100.0),
                     _ep("Google", "google-vertex/global", "0.30", "2.50", 1048576, 65536, "max_tokens seed response_format structured_outputs reasoning stop", 100.0),
                     _ep("Google", "google-vertex/eu", "0.33", "2.75", 1048576, 65536, "max_tokens seed response_format structured_outputs reasoning stop", 100.0),
                     _ep("Google", "google-vertex/us", "0.33", "2.75", 1048576, 65536, "max_tokens seed response_format structured_outputs reasoning stop", 100.0),
                     _ep("Google", "google-vertex/global/priority", "0.54", "4.50", 1048576, 65536, "max_tokens seed response_format structured_outputs reasoning stop", 99.85, "priority tier"),
                     _ep("Google AI Studio", "google-ai-studio/priority", "0.54", "4.50", 1048576, 65536, "temperature top_p max_tokens seed response_format structured_outputs reasoning", 100.0, "priority tier")]
_ENDPOINTS_QWEN = [_ep("Darkbloom", "darkbloom/fp4", "0.065", "0.75", 262144, 32768, _STD + " max_tokens reasoning", 98.23, "fp4; 30-minute uptime 77.78"),
                   _ep("DeepInfra", "deepinfra/fp8", "0.14", "1.00", 262144, 81920, _STD + " max_tokens reasoning", 99.26),
                   _ep("Parasail", "parasail/fp8", "0.15", "1.00", 262144, 235929, _STD + " max_tokens reasoning", 100.0),
                   _ep("Venice", "venice", "0.15", "1.00", 256000, 16384, "temperature response_format structured_outputs max_tokens reasoning", 100.0, "no seed"),
                   _ep("Alibaba", "alibaba", "0.1625", "1.30", 262144, 65536, _STD + " max_tokens reasoning", 100.0, "datacenters SG, CN per OpenRouter providers metadata"),
                   _ep("AtlasCloud", "atlas-cloud/fp8", "0.225", "1.80", 262144, 65536, _STD + " max_tokens reasoning", 99.08),
                   _ep("SiliconFlow", "siliconflow/fp8", "0.24", "1.80", 262144, 235929, "temperature response_format structured_outputs max_tokens reasoning", 95.65, "no seed")]
_ENDPOINTS_DEEPSEEK = [_ep("DeepInfra", "deepinfra/fp4", "0.25", "0.95", 163840, 32768, _STD + " max_tokens", 100.0, "fp4 quantisation"),
                       _ep("SiliconFlow", "siliconflow/fp8", "0.27", "1.00", 163840, 147456, "temperature response_format structured_outputs max_tokens", 97.85, "no seed"),
                       _ep("CoreWeave", "coreweave/fp8", "0.55", "1.65", 161000, 144900, _STD + " max_tokens", 100.0),
                       _ep("Mara", "mara", "0.60", "1.70", 131072, 117964, "temperature response_format structured_outputs max_tokens", 98.40, "no seed"),
                       _ep("SambaNova", "sambanova/fp8", "0.65", "1.50", 131072, 7168, "temperature max_tokens", 97.33, "no response_format; max completion 7,168")]
_ENDPOINTS_MISTRAL = [_ep("DeepInfra", "deepinfra/fp8", "0.075", "0.20", 128000, 16384, _STD + " max_tokens", 99.86),
                      _ep("Parasail", "parasail/bf16", "0.09", "0.30", 131072, 32768, _STD + " max_tokens", 94.31),
                      _ep("Venice", "venice/fp8", "0.09375", "0.25", 256000, 16384, "temperature response_format structured_outputs max_tokens", 98.19, "no seed"),
                      _ep("Mistral", "mistral/eu", "0.10", "0.30", 32768, 26214, _STD + " max_tokens", 100.0, "first-party vendor; context 32,768")]
_SONNET_COMMON = "max_tokens stop response_format reasoning include_reasoning verbosity reasoning_effort tools tool_choice"
_ENDPOINTS_SONNET55 = [_ep("Anthropic", "anthropic", "2.00", "10.00", 1000000, 128000, _SONNET_COMMON + " structured_outputs", 100.0),
                       _ep("Google", "google-vertex/global", "2.00", "10.00", 1000000, 128000, _SONNET_COMMON + " structured_outputs", 100.0),
                       _ep("Amazon Bedrock", "amazon-bedrock", "2.00", "10.00", 1000000, 128000, _SONNET_COMMON, 100.0),
                       _ep("Azure", "azure/global", "2.00", "10.00", 1000000, 128000, _SONNET_COMMON + " structured_outputs max_completion_tokens", 96.25),
                       _ep("Claude Platform on AWS", "claude-on-aws", "2.00", "10.00", 1000000, 128000, _SONNET_COMMON, 100.0)]

_JEV_ENDPOINT_NOTE = "OpenRouter endpoints metadata lists one endpoint named `typesafe/jev-1.14-20261014`, so a dated snapshot of this form exists for 1.14."

ROSTER: List[ModelRecord] = [
    _chat_record("baseline-intent-gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite (production baseline; benchmark route)", ["PRODUCTION_BASELINE"], ["INTENT"],
                 "google/gemini-3.5-flash-lite", _ENDPOINTS_GEMINI, "google-ai-studio", production_name="Gemini 3.5 Flash Lite",
                 extra_decisions=["The Dify deployment (provider, tier, settings) is unknown; this is the standard-tier Google AI Studio route chosen because it supports temperature. "
                                  "Results are labelled benchmark-route, not production.",
                                  "The legacy catalog lists $0.075/$0.30 for this model; OpenRouter lists $0.30/$2.50 (standard) and $0.15/$1.25 (flex). The registry is authoritative."]),
    _chat_record("baseline-lead-gpt-4.1-mini", "GPT-4.1-mini (production baseline; benchmark route)", ["PRODUCTION_BASELINE"], ["LEAD"],
                 "openai/gpt-4.1-mini", _ENDPOINTS_41MINI, "openai", production_name="GPT-4.1-mini",
                 extra_decisions=["The Dify deployment (provider, settings) is unknown; results are labelled benchmark-route, not production."]),
    JEV_1_14,
    _chat_record("gpt-4.1-nano", "GPT-4.1 Nano (OpenAI)", ["CANDIDATE"], ["INTENT", "LEAD"], "openai/gpt-4.1-nano", _ENDPOINTS_NANO, "openai"),
    _chat_record("gpt-4o-mini", "GPT-4o mini (OpenAI)", ["CANDIDATE"], ["INTENT", "LEAD"], "openai/gpt-4o-mini", _ENDPOINTS_4OMINI, "openai"),
    _chat_record("qwen3.5-35b-a3b", "Qwen3.5 35B A3B", ["CANDIDATE"], ["INTENT", "LEAD"], "qwen/qwen3.5-35b-a3b", _ENDPOINTS_QWEN, "deepinfra/fp8",
                 extra_decisions=["Seven providers are listed; DeepInfra fp8 (US, 100% 30-minute uptime) is proposed. Alibaba (SG/CN datacenters) is the vendor route. Choose before the pilot."]),
    _chat_record("deepseek-chat-v3.1", "DeepSeek Chat v3.1", ["CANDIDATE"], ["INTENT", "LEAD"], "deepseek/deepseek-chat-v3.1", _ENDPOINTS_DEEPSEEK, "coreweave/fp8",
                 extra_decisions=["CoreWeave fp8 is proposed to avoid 4-bit quantisation (DeepInfra is fp4) and the 7,168-token SambaNova completion limit. DeepSeek's own route is not listed."]),
    _chat_record("mistral-small-3.2-24b-instruct", "Mistral Small 3.2 24B Instruct", ["CANDIDATE"], ["INTENT", "LEAD"], "mistralai/mistral-small-3.2-24b-instruct", _ENDPOINTS_MISTRAL, "mistral/eu",
                 extra_decisions=["The first-party `mistral/eu` route is proposed; its 32,768-token context is enough for the prompts and states used here but is the smallest in the roster."]),
    _chat_record("gpt-4.1", "GPT-4.1 (frontier reference)", ["FRONTIER_REFERENCE"], ["INTENT", "LEAD"], "openai/gpt-4.1", _ENDPOINTS_41, "openai"),
    _chat_record("claude-sonnet-5.5", "Claude Sonnet 5.5 (frontier reference)", ["FRONTIER_REFERENCE"], ["INTENT", "LEAD"], "anthropic/claude-sonnet-5.5",
                 _ENDPOINTS_SONNET55, "anthropic", version_note="Endpoint names carry the fixed snapshot 20260928; the dated slug resolves to the same listing but direct requestability is unverified.",
                 extra_assumptions=["Latest Claude Sonnet found on OpenRouter on 2026-10-15 (created Sept 2026); Sonnet 5 (snapshot 20260630) is the previous version."],
                 extra_decisions=["Use `anthropic/claude-sonnet-5.5` and record the resolved snapshot (expected 20260928) in every record; do not use the moving alias ~anthropic/claude-sonnet-latest.",
                                  "Temperature is not listed for this model's endpoints, so it is omitted; outputs may therefore not be deterministic."]),
]
REGISTRY: Dict[str, ModelRecord] = {r.key: r for r in ROSTER}


MODEL_ID_INDEX: Dict[str, str] = {r.requested_model_id.value: r.key for r in REGISTRY.values() if r.requested_model_id.value}


def record_for_model_id(model_id: str) -> Optional[ModelRecord]:
    key = MODEL_ID_INDEX.get(model_id)
    return REGISTRY[key] if key else None


def synthetic_live_blockers(model_id: str, stage: str = "pilot") -> List[str]:
    """Blockers for a synthetic-only live call. Models without a registry record are unrestricted here (legacy catalog)."""
    rec = record_for_model_id(model_id)
    return rec.synthetic_blockers(stage) if rec else []


def roster_status() -> List[Dict[str, Any]]:
    """One row per participant: readiness (BLOCKED / SMOKE_READY / READY), the exact blockers, and evaluation state."""
    rows = []
    for r in REGISTRY.values():
        strict, smoke = r.synthetic_blockers("pilot"), r.synthetic_blockers("smoke")
        readiness = "READY" if not strict else ("SMOKE_READY" if not smoke else "BLOCKED")
        rows.append({
            "key": r.key, "display_name": r.display_name, "roles": r.roles, "tasks": r.tasks, "protocol": r.protocol.value,
            "model_id": r.requested_model_id.value, "route_status": r.route.status.value, "readiness": readiness,
            "blockers": smoke if readiness == "BLOCKED" else strict, "student_data_blockers": r.student_data_blockers(),
            "data_policy": r.data_policy.status.value, "price_status": r.input_price_per_1m_usd.status.value,
            "evaluation": "NOT_YET_EVALUATED",
            "pinned_route": r.pinned_tag, "route_price": (f"${r.route.route_input_price_per_1m_usd.value}/${r.route.route_output_price_per_1m_usd.value} per 1M"
                                                           if r.route.route_input_price_per_1m_usd.value is not None else None),
            "source": (r.requested_model_id.source_urls[0] if r.requested_model_id.source_urls else "owner-supplied"),
            "checked_on": str(r.requested_model_id.checked_on) if r.requested_model_id.checked_on else None,
            "production_reported_name": r.production_reported_name,
            "benchmark_route_only": bool(r.production_reported_name),
        })
    return rows


def model_level_prices(model_id: str) -> Optional[Tuple[Decimal, Decimal]]:
    """(input, output) USD per 1M tokens for budgeting preflight, from VERIFIED_PRIMARY or USER_SUPPLIED facts only."""
    rec = record_for_model_id(model_id)
    if rec is None:
        return None
    ok = (FactStatus.VERIFIED_PRIMARY, FactStatus.USER_SUPPLIED)
    a, b = rec.input_price_per_1m_usd, rec.output_price_per_1m_usd
    if a.status in ok and b.status in ok:
        return Decimal(str(a.value)), Decimal(str(b.value))
    return None


def estimate_permitted(model_id: str) -> bool:
    """A catalog cost estimate needs a VERIFIED route-specific price. Legacy models without a record are unchanged."""
    rec = record_for_model_id(model_id)
    if rec is None:
        return True
    r = rec.route
    return (r.status == RouteStatus.READY and r.route_input_price_per_1m_usd.status == FactStatus.VERIFIED_PRIMARY
            and r.route_output_price_per_1m_usd.status == FactStatus.VERIFIED_PRIMARY)


OPENROUTER_COST_UNIT = unknown(
    "OpenRouter's usage-accounting docs call `usage.cost` 'cost in credits' and never define a credit; its contact page says credits are USD-denominated "
    "but no 1:1 ratio is stated (https://openrouter.ai/docs/guides/guides/usage-accounting). Until the ratio is verified (for example by comparing a "
    "returned cost with the dashboard balance), the native value is preserved but never booked as USD."
)
NATIVE_COST_UNIT_LABEL = "OPENROUTER_CREDITS_USD_RATE_UNVERIFIED"


def provider_cost_is_usd() -> bool:
    """True only when the unit of provider-reported `usage.cost` has been verified as USD."""
    return OPENROUTER_COST_UNIT.status == FactStatus.VERIFIED_PRIMARY


def get_record(key: str) -> ModelRecord:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"No sourced registry record for '{key}'")


def live_blockers(key: str) -> List[str]:
    return get_record(key).readiness_blockers()
