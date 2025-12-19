"""Pydantic models and helpers for the financial modeling web application."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


def generate_id() -> UUID:
    return uuid4()


# Core entities
class DimensionMember(BaseModel):
    code: str
    name: str
    parent_code: Optional[str] = None


class DimensionDefinition(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    name: str
    type: str = Field(description="hierarchical|attribute|currency")
    members: List[DimensionMember] = Field(default_factory=list)

    @validator("type")
    def validate_type(cls, value: str) -> str:
        allowed = {"hierarchical", "attribute", "currency"}
        if value not in allowed:
            raise ValueError(f"Dimension type must be one of {', '.join(sorted(allowed))}")
        return value


class ModelDefinition(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    name: str
    description: Optional[str] = None
    owner: str
    base_currency: str = "USD"
    fiscal_calendar: str = "Jan-Dec"
    status: str = "draft"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    dimensions: List[DimensionDefinition] = Field(default_factory=list)


class ScenarioDefinition(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    model_id: UUID
    name: str
    parent_scenario_id: Optional[UUID] = None
    description: Optional[str] = None
    status: str = "draft"
    color: str = "#6c43f3"


class DriverDefinition(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    model_id: UUID
    code: str
    name: str
    description: Optional[str] = None
    data_type: str = "number"
    default_value: float = 0.0
    dimensionality: List[str] = Field(default_factory=list)

    @validator("code")
    def validate_code(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Driver code must not be empty")
        return value


class DriverValue(BaseModel):
    driver_id: UUID
    scenario_id: UUID
    period: str
    dimension_slice: Dict[str, str] = Field(default_factory=dict)
    value: float
    source: str = "manual"

    def slice_key(self) -> Tuple[str, ...]:
        return tuple(sorted(f"{k}:{v}" for k, v in self.dimension_slice.items()))


class FormulaDefinition(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    model_id: UUID
    target_measure: str
    scope: List[str] = Field(default_factory=list, description="Dimensions that affect the formula")
    expression: str
    dependencies: List[str] = Field(default_factory=list)

    @validator("target_measure")
    def normalize_measure(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Target measure name must not be empty")
        return value


class FxRate(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    date: date
    from_currency: str
    to_currency: str
    average_rate: float
    closing_rate: float


class VersionRecord(BaseModel):
    id: UUID = Field(default_factory=generate_id)
    model_id: UUID
    scenario_id: UUID
    tag: str
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "draft"
    snapshot: Dict[str, Any] = Field(default_factory=dict)


# API schemas
class ModelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    owner: str
    base_currency: str = "USD"
    fiscal_calendar: str = "Jan-Dec"


class ScenarioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    parent_scenario_id: Optional[UUID] = None
    status: str = "draft"
    color: str = "#29967f"


class DimensionCreate(BaseModel):
    name: str
    type: str = "hierarchical"
    members: List[DimensionMember] = Field(default_factory=list)


class DriverCreate(BaseModel):
    code: str
    name: str
    description: Optional[str] = None
    data_type: str = "number"
    default_value: float = 0.0
    dimensionality: List[str] = Field(default_factory=list)


class DriverValueUpsert(BaseModel):
    driver_id: UUID
    period: str
    dimension_slice: Dict[str, str] = Field(default_factory=dict)
    value: float
    source: str = "manual"


class FormulaCreate(BaseModel):
    target_measure: str
    scope: List[str] = Field(default_factory=list)
    expression: str


class CalculationRequest(BaseModel):
    scenario_id: UUID
    period: str
    dimension_slice: Dict[str, str] = Field(default_factory=dict)
    target_currency: Optional[str] = None


class VersionCreate(BaseModel):
    scenario_id: UUID
    tag: str
    created_by: str


class ReportRequest(BaseModel):
    scenario_id: UUID
    period: str
    measures: List[str]
    dimension_slice: Dict[str, str] = Field(default_factory=dict)


class StatementRow(BaseModel):
    measure: str
    value: float
    period: str
    dimension_slice: Dict[str, str] = Field(default_factory=dict)
    currency: str


def extract_identifiers(expression: str) -> Iterable[str]:
    """Return identifier-like tokens from an expression to approximate dependencies."""
    import ast

    tree = ast.parse(expression, mode="eval")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names

