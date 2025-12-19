"""FastAPI application exposing the financial modeling prototype."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
from uuid import UUID

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    CalculationRequest,
    DimensionCreate,
    DriverCreate,
    DriverValueUpsert,
    FormulaCreate,
    ModelCreate,
    ReportRequest,
    ScenarioCreate,
    VersionCreate,
)
from .services import CalculationResult, seed_demo_data, store

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Plataforma de Modelagem Financeira",
        description="Aplicação web modular para modelagem financeira multidimensional.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=FileResponse)
    def root() -> Path:
        return static_dir / "index.html"

    @app.post("/models")
    def create_model(payload: ModelCreate):
        model = store.create_model(payload)
        return model

    @app.get("/models")
    def list_models():
        return list(store.models.values())

    @app.post("/models/{model_id}/dimensions")
    def add_dimension(model_id: UUID, payload: DimensionCreate):
        if model_id not in store.models:
            raise HTTPException(status_code=404, detail="Model not found")
        return store.add_dimension(model_id, payload)

    @app.get("/models/{model_id}/dimensions")
    def list_dimensions(model_id: UUID):
        return store.list_dimensions(model_id)

    @app.post("/models/{model_id}/scenarios")
    def create_scenario(model_id: UUID, payload: ScenarioCreate):
        if model_id not in store.models:
            raise HTTPException(status_code=404, detail="Model not found")
        return store.create_scenario(model_id, payload)

    @app.get("/models/{model_id}/scenarios")
    def list_scenarios(model_id: UUID):
        return store.list_scenarios(model_id)

    @app.post("/models/{model_id}/drivers")
    def create_driver(model_id: UUID, payload: DriverCreate):
        if model_id not in store.models:
            raise HTTPException(status_code=404, detail="Model not found")
        return store.create_driver(model_id, payload)

    @app.get("/models/{model_id}/drivers")
    def list_drivers(model_id: UUID):
        return store.list_drivers(model_id)

    @app.post("/models/{model_id}/driver-values/{scenario_id}")
    def upsert_driver_values(model_id: UUID, scenario_id: UUID, payload: List[DriverValueUpsert]):
        try:
            return store.upsert_driver_values(model_id, scenario_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/{model_id}/formulas")
    def create_formula(model_id: UUID, payload: FormulaCreate):
        if model_id not in store.models:
            raise HTTPException(status_code=404, detail="Model not found")
        return store.create_formula(model_id, payload)

    @app.get("/models/{model_id}/formulas")
    def list_formulas(model_id: UUID):
        return store.list_formulas(model_id)

    @app.post("/models/{model_id}/calculate")
    def calculate(model_id: UUID, payload: CalculationRequest):
        try:
            result: CalculationResult = store.calculate(model_id, payload)
            return JSONResponse({"currency": result.currency, "measure_values": result.measure_values})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/models/{model_id}/versions")
    def create_version(model_id: UUID, payload: VersionCreate):
        try:
            return store.create_version(model_id, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/models/{model_id}/versions")
    def list_versions(model_id: UUID):
        return store.list_versions(model_id)

    @app.post("/models/{model_id}/reports/statement")
    def render_statement(model_id: UUID, payload: ReportRequest):
        return store.render_report(model_id, payload)

    return app


app = create_app()
seed_demo_data()


if __name__ == "__main__":
    uvicorn.run("src.webapp.app:app", host="0.0.0.0", port=8000, reload=True)
