"""Service layer with in-memory storage for the financial modeling web app."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Set, Tuple
from uuid import UUID

from .models import (
    CalculationRequest,
    DimensionCreate,
    DimensionDefinition,
    DimensionMember,
    DriverCreate,
    DriverDefinition,
    DriverValue,
    DriverValueUpsert,
    FxRate,
    FormulaCreate,
    FormulaDefinition,
    ModelCreate,
    ModelDefinition,
    ReportRequest,
    ScenarioCreate,
    ScenarioDefinition,
    StatementRow,
    VersionCreate,
    VersionRecord,
    extract_identifiers,
    generate_id,
)


class SafeEvaluator:
    """A small expression evaluator used by the calculation engine."""

    allowed_nodes: Tuple[type, ...] = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.Call,
        ast.Compare,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
        ast.USub,
        ast.UAdd,
    )

    allowed_functions = {
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
    }

    def __call__(self, expression: str, context: Dict[str, float]) -> float:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, self.allowed_nodes):
                raise ValueError(f"Operation not allowed in expression: {type(node).__name__}")
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, "id", None)
                if func_name not in self.allowed_functions:
                    raise ValueError(f"Function {func_name} not permitted in expressions.")
        compiled = compile(tree, "<expression>", "eval")
        return float(
            eval(
                compiled,
                {"__builtins__": {}},
                {**self.allowed_functions, **context},
            )
        )


@dataclass
class CalculationResult:
    measure_values: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"


class InMemoryStore:
    """Stores application data and orchestrates calculations."""

    def __init__(self) -> None:
        self.models: Dict[UUID, ModelDefinition] = {}
        self.scenarios: Dict[UUID, ScenarioDefinition] = {}
        self.dimensions: Dict[UUID, List[DimensionDefinition]] = defaultdict(list)
        self.drivers: Dict[UUID, DriverDefinition] = {}
        self.driver_values: Dict[Tuple[UUID, UUID, str, Tuple[str, ...]], DriverValue] = {}
        self.formulas: Dict[UUID, FormulaDefinition] = {}
        self.fx_rates: List[FxRate] = []
        self.versions: Dict[UUID, VersionRecord] = {}
        self.evaluator = SafeEvaluator()

    # Model + dimension management
    def create_model(self, payload: ModelCreate) -> ModelDefinition:
        model = ModelDefinition(
            name=payload.name,
            description=payload.description,
            owner=payload.owner,
            base_currency=payload.base_currency,
            fiscal_calendar=payload.fiscal_calendar,
        )
        self.models[model.id] = model
        return model

    def add_dimension(self, model_id: UUID, payload: DimensionCreate) -> DimensionDefinition:
        dimension = DimensionDefinition(name=payload.name, type=payload.type, members=payload.members)
        self.dimensions[model_id].append(dimension)
        return dimension

    def list_dimensions(self, model_id: UUID) -> List[DimensionDefinition]:
        return self.dimensions.get(model_id, [])

    # Scenario
    def create_scenario(self, model_id: UUID, payload: ScenarioCreate) -> ScenarioDefinition:
        scenario = ScenarioDefinition(
            model_id=model_id,
            name=payload.name,
            parent_scenario_id=payload.parent_scenario_id,
            description=payload.description,
            status=payload.status,
            color=payload.color,
        )
        self.scenarios[scenario.id] = scenario
        return scenario

    def list_scenarios(self, model_id: UUID) -> List[ScenarioDefinition]:
        return [s for s in self.scenarios.values() if s.model_id == model_id]

    # Drivers
    def create_driver(self, model_id: UUID, payload: DriverCreate) -> DriverDefinition:
        driver = DriverDefinition(
            model_id=model_id,
            code=payload.code,
            name=payload.name,
            description=payload.description,
            data_type=payload.data_type,
            default_value=payload.default_value,
            dimensionality=payload.dimensionality,
        )
        self.drivers[driver.id] = driver
        return driver

    def list_drivers(self, model_id: UUID) -> List[DriverDefinition]:
        return [d for d in self.drivers.values() if d.model_id == model_id]

    def upsert_driver_values(self, model_id: UUID, scenario_id: UUID, payloads: List[DriverValueUpsert]) -> List[DriverValue]:
        saved: List[DriverValue] = []
        for payload in payloads:
            if payload.driver_id not in self.drivers:
                raise ValueError(f"Driver {payload.driver_id} not found.")
            dv = DriverValue(
                driver_id=payload.driver_id,
                scenario_id=scenario_id,
                period=payload.period,
                dimension_slice=payload.dimension_slice,
                value=payload.value,
                source=payload.source,
            )
            key = (payload.driver_id, scenario_id, payload.period, dv.slice_key())
            self.driver_values[key] = dv
            saved.append(dv)
        return saved

    def get_driver_value(
        self, driver_id: UUID, scenario_id: UUID, period: str, slice_key: Tuple[str, ...]
    ) -> Optional[DriverValue]:
        scenario = self.scenarios[scenario_id]
        lookup_key = (driver_id, scenario_id, period, slice_key)
        if lookup_key in self.driver_values:
            return self.driver_values[lookup_key]
        if scenario.parent_scenario_id:
            return self.get_driver_value(driver_id, scenario.parent_scenario_id, period, slice_key)
        return None

    # Formulas
    def create_formula(self, model_id: UUID, payload: FormulaCreate) -> FormulaDefinition:
        dependencies = list(extract_identifiers(payload.expression))
        formula = FormulaDefinition(
            model_id=model_id,
            target_measure=payload.target_measure,
            scope=payload.scope,
            expression=payload.expression,
            dependencies=dependencies,
        )
        self.formulas[formula.id] = formula
        return formula

    def list_formulas(self, model_id: UUID) -> List[FormulaDefinition]:
        return [f for f in self.formulas.values() if f.model_id == model_id]

    # FX rates
    def set_fx_rate(self, rate: FxRate) -> FxRate:
        self.fx_rates.append(rate)
        return rate

    def get_fx_rate(self, from_currency: str, to_currency: str, rate_date: date, rate_type: str) -> Optional[FxRate]:
        for rate in sorted(self.fx_rates, key=lambda r: r.date, reverse=True):
            if (
                rate.from_currency == from_currency
                and rate.to_currency == to_currency
                and rate.date <= rate_date
            ):
                return rate
        return None

    # Calculation
    def _topologically_sorted_formulas(self, model_id: UUID) -> List[FormulaDefinition]:
        formulas = self.list_formulas(model_id)
        measure_lookup: Dict[str, FormulaDefinition] = {f.target_measure: f for f in formulas}
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        in_degrees: Dict[str, int] = {f.target_measure: 0 for f in formulas}

        for formula in formulas:
            for dep in formula.dependencies:
                if dep in measure_lookup:
                    adjacency[dep].add(formula.target_measure)
                    in_degrees[formula.target_measure] += 1

        queue = deque([measure for measure, degree in in_degrees.items() if degree == 0])
        ordered: List[FormulaDefinition] = []
        while queue:
            measure = queue.popleft()
            ordered.append(measure_lookup[measure])
            for dependent in adjacency.get(measure, []):
                in_degrees[dependent] -= 1
                if in_degrees[dependent] == 0:
                    queue.append(dependent)

        if len(ordered) != len(formulas):
            raise ValueError("Detected a circular dependency between formulas.")
        return ordered

    def _build_context(
        self, model_id: UUID, scenario_id: UUID, period: str, slice_key: Tuple[str, ...], existing: Dict[str, float]
    ) -> Dict[str, float]:
        context = dict(existing)
        for driver in self.list_drivers(model_id):
            dv = self.get_driver_value(driver.id, scenario_id, period, slice_key)
            context[driver.code] = dv.value if dv else driver.default_value
        context.update(existing)
        return context

    def calculate(self, model_id: UUID, payload: CalculationRequest) -> CalculationResult:
        model = self.models[model_id]
        slice_key = tuple(sorted(f"{k}:{v}" for k, v in payload.dimension_slice.items()))
        ordered_formulas = self._topologically_sorted_formulas(model_id)
        context: Dict[str, float] = {}
        for formula in ordered_formulas:
            local_context = self._build_context(model_id, payload.scenario_id, payload.period, slice_key, context)
            result = self.evaluator(formula.expression, local_context)
            context[formula.target_measure] = result
        currency = payload.target_currency or model.base_currency
        return CalculationResult(measure_values=context, currency=currency)

    # Versioning
    def create_version(self, model_id: UUID, payload: VersionCreate) -> VersionRecord:
        result = self.calculate(
            model_id,
            CalculationRequest(
                scenario_id=payload.scenario_id,
                period="latest",
                dimension_slice={},
            ),
        )
        version = VersionRecord(
            model_id=model_id,
            scenario_id=payload.scenario_id,
            tag=payload.tag,
            created_by=payload.created_by,
            snapshot=result.measure_values,
        )
        self.versions[version.id] = version
        return version

    def list_versions(self, model_id: UUID) -> List[VersionRecord]:
        return [v for v in self.versions.values() if v.model_id == model_id]

    # Reporting
    def render_report(self, model_id: UUID, payload: ReportRequest) -> List[StatementRow]:
        calc = self.calculate(
            model_id,
            CalculationRequest(
                scenario_id=payload.scenario_id,
                period=payload.period,
                dimension_slice=payload.dimension_slice,
                target_currency=None,
            ),
        )
        rows = [
            StatementRow(
                measure=measure,
                value=calc.measure_values.get(measure, 0.0),
                period=payload.period,
                dimension_slice=payload.dimension_slice,
                currency=calc.currency,
            )
            for measure in payload.measures
        ]
        return rows


store = InMemoryStore()


def seed_demo_data() -> None:
    """Preload the in-memory store with an opinionated demo dataset."""
    base_model = store.create_model(
        ModelCreate(
            name="Corporate Plan",
            description="Modelo multidimensional com demonstrações integradas",
            owner="finance-team",
            base_currency="USD",
            fiscal_calendar="Jan-Dec",
        )
    )
    entity_dimension = store.add_dimension(
        base_model.id,
        DimensionCreate(
            name="Entity",
            type="hierarchical",
            members=[
                DimensionMember(code="GRP", name="Holding Group"),
                DimensionMember(code="RET", name="Retail Co", parent_code="GRP"),
                DimensionMember(code="WHO", name="Wholesale Co", parent_code="GRP"),
            ],
        ),
    )
    store.add_dimension(
        base_model.id,
        DimensionCreate(
            name="Product",
            type="attribute",
            members=[DimensionMember(code="P1", name="Core"), DimensionMember(code="P2", name="Premium")],
        ),
    )
    baseline = store.create_scenario(
        base_model.id,
        ScenarioCreate(name="Baseline", description="Cenário neutro para FY26", status="active", color="#0f766e"),
    )
    optimistic = store.create_scenario(
        base_model.id,
        ScenarioCreate(
            name="Otimista",
            description="Ajuste de volume e preço com câmbio favorável",
            parent_scenario_id=baseline.id,
            status="draft",
            color="#06b6d4",
        ),
    )
    price_driver = store.create_driver(
        base_model.id,
        DriverCreate(
            code="price",
            name="Preço Médio",
            description="Preço médio por unidade",
            default_value=120.0,
            dimensionality=["Product"],
        ),
    )
    volume_driver = store.create_driver(
        base_model.id,
        DriverCreate(
            code="volume",
            name="Volume Vendido",
            description="Unidades vendidas por período",
            default_value=10_000.0,
            dimensionality=["Entity", "Product"],
        ),
    )
    cost_driver = store.create_driver(
        base_model.id,
        DriverCreate(
            code="unit_cost",
            name="Custo Unitário",
            description="Custo médio por unidade",
            default_value=60.0,
            dimensionality=["Product"],
        ),
    )
    # Override optimistic scenario with higher volume
    store.upsert_driver_values(
        base_model.id,
        optimistic.id,
        [
            DriverValueUpsert(
                driver_id=volume_driver.id,
                period="2026-01",
                dimension_slice={"Entity": "RET", "Product": "P2"},
                value=12_500.0,
            )
        ],
    )
    # Formulas
    store.create_formula(
        base_model.id,
        FormulaCreate(
            target_measure="revenue",
            expression="price * volume",
            scope=["Entity", "Product"],
        ),
    )
    store.create_formula(
        base_model.id,
        FormulaCreate(
            target_measure="cogs",
            expression="unit_cost * volume",
            scope=["Entity", "Product"],
        ),
    )
    store.create_formula(
        base_model.id,
        FormulaCreate(
            target_measure="gross_profit",
            expression="revenue - cogs",
            scope=["Entity", "Product"],
        ),
    )
    store.set_fx_rate(
        FxRate(
            date=date(2025, 12, 31),
            from_currency="USD",
            to_currency="BRL",
            average_rate=5.0,
            closing_rate=5.2,
        )
    )
