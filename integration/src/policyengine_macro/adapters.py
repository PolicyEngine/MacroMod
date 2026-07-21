"""New-model adapter contract built on the production ``ScoreResult``.

The common request is capability-checked before execution.  The common result
is the same object already emitted by scoring entry points; adapters may not
introduce a second result envelope or infer units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from policyengine_macro.capabilities import get_status
from policyengine_macro.core import ScoreResult, obr_score_reform


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    analysis_type: str = Field(min_length=1)
    country: str = Field(pattern=r"^[a-z]{2}$")
    inputs: dict[str, Any]
    baseline: str = Field(min_length=1)
    horizon: str = Field(min_length=1)
    requested_outputs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def registered_and_supported(self) -> "AnalysisRequest":
        capability = get_status(self.model_id)
        if self.country not in capability["geography"]:
            raise ValueError(f"{self.model_id} does not support {self.country}")
        if self.analysis_type not in capability["question_types"]:
            raise ValueError(f"{self.model_id} does not support {self.analysis_type}")
        unsupported = sorted(set(self.requested_outputs) - set(capability["outputs"]))
        if unsupported:
            raise ValueError(f"unsupported outputs for {self.model_id}: {unsupported}")
        return self


class ModelAdapter(ABC):
    """Base class and fail-closed request/result boundary for model packages."""

    model_id: str

    @abstractmethod
    def validate_request(self, request: AnalysisRequest) -> None:
        """Validate names, ranges, dimensions, frequency and units."""

    @abstractmethod
    def execute(self, request: AnalysisRequest) -> ScoreResult | dict:
        """Execute the model and return the canonical result contract."""

    def run(self, request: AnalysisRequest | dict) -> ScoreResult:
        request = AnalysisRequest.model_validate(request)
        if request.model_id != self.model_id:
            raise ValueError(
                f"adapter {self.model_id} cannot run request for {request.model_id}"
            )
        self.validate_request(request)
        result = ScoreResult.model_validate(self.execute(request))
        mismatches = []
        if result.model != request.model_id:
            mismatches.append("model")
        if result.country != request.country:
            mismatches.append("country")
        if result.baseline != request.baseline:
            mismatches.append("baseline")
        if result.horizon != request.horizon:
            mismatches.append("horizon")
        missing = set(request.requested_outputs) - set(result.quantities)
        if missing:
            mismatches.append(f"outputs {sorted(missing)}")
        if mismatches:
            raise ValueError(
                "adapter result does not match request: " + ", ".join(mismatches)
            )
        return result


class OBRPolicyReformAdapter(ModelAdapter):
    """First real contract implementation: reviewed reform-to-OBR bridge."""

    model_id = "obr-macro"
    allowed_inputs = {"reform", "start_year", "years", "dataset"}

    def validate_request(self, request: AnalysisRequest) -> None:
        unknown = set(request.inputs) - self.allowed_inputs
        if unknown:
            raise ValueError(f"unknown OBR adapter inputs: {sorted(unknown)}")
        if "reform" not in request.inputs:
            raise ValueError("OBR adapter requires an explicit reform mapping")
        if int(request.inputs.get("years", 5)) < 1:
            raise ValueError("years must be at least 1")

    def execute(self, request: AnalysisRequest) -> ScoreResult:
        payload = obr_score_reform(**request.inputs)
        return ScoreResult.model_validate(payload["score"])

