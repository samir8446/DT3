"""Streaming digital-twin service (deployment path).

A battery system (BMS, PLC, historian or MQTT bridge) posts one discharge cycle at a time; the service keeps
one StreamingTwin per battery and returns the updated state, remaining-life forecast, dominant degradation
mechanism and alarms.

Run:      pip install fastapi uvicorn && uvicorn service:app --host 0.0.0.0 --port 8000
Example:  curl -X POST localhost:8000/batteries/B0005/cycles -H 'content-type: application/json' \\
               -d '{"capacity_Ah": 1.85, "T_C": 32.1, "I_A": 2.0}'

The registry below has no web dependency, so it can also be embedded in an OPC UA / MQTT gateway or tested
directly.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import twin_engine as te


@dataclass
class BatteryConfig:
    c_bol_Ah: float = 2.0
    eol_soh: float = 0.7
    horizon: int = 300


@dataclass
class TwinRegistry:
    """Thread-safe collection of streaming twins, one per battery id."""
    default: BatteryConfig = field(default_factory=BatteryConfig)
    twins: Dict[str, te.StreamingTwin] = field(default_factory=dict)
    history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, battery_id: str, cfg: Optional[BatteryConfig] = None) -> None:
        cfg = cfg or self.default
        with self._lock:
            self.twins[battery_id] = te.StreamingTwin(cfg.c_bol_Ah, cfg.eol_soh, cfg.horizon)
            self.history[battery_id] = []

    def ingest(self, battery_id: str, capacity_Ah: float, T_C: float, I_A: float,
               R_ohm: Optional[float] = None) -> Dict[str, Any]:
        if not (capacity_Ah > 0 and I_A >= 0 and -40 <= T_C <= 90):
            raise ValueError("capacity_Ah must be > 0, I_A >= 0 and T_C within -40..90 °C")
        if battery_id not in self.twins:
            self.register(battery_id)
        with self._lock:
            state = self.twins[battery_id].ingest(capacity_Ah, T_C, I_A, R_ohm)
            self.history[battery_id].append(state)
        state["battery_id"] = battery_id
        state["engine_version"] = te.ENGINE_VERSION
        return state

    def state(self, battery_id: str) -> Dict[str, Any]:
        h = self.history.get(battery_id)
        if not h:
            raise KeyError(battery_id)
        return h[-1]

    def fleet(self) -> List[Dict[str, Any]]:
        return [{"battery_id": b, **{k: h[-1].get(k) for k in ("cycle", "SOH_measured", "RUL_cycles",
                                                                "dominant_mechanism", "alarms")}}
                for b, h in self.history.items() if h]


REGISTRY = TwinRegistry()

try:                                          # optional web layer
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class CycleIn(BaseModel):
        capacity_Ah: float
        T_C: float
        I_A: float
        R_ohm: Optional[float] = None

    class BatteryIn(BaseModel):
        c_bol_Ah: float = 2.0
        eol_soh: float = 0.7
        horizon: int = 300

    app = FastAPI(title="Battery digital twin", version=te.ENGINE_VERSION)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "engine_version": te.ENGINE_VERSION, "batteries": len(REGISTRY.twins)}

    @app.put("/batteries/{battery_id}")
    def register(battery_id: str, cfg: BatteryIn) -> Dict[str, str]:
        REGISTRY.register(battery_id, BatteryConfig(cfg.c_bol_Ah, cfg.eol_soh, cfg.horizon))
        return {"registered": battery_id}

    @app.post("/batteries/{battery_id}/cycles")
    def ingest(battery_id: str, c: CycleIn) -> Dict[str, Any]:
        try:
            return REGISTRY.ingest(battery_id, c.capacity_Ah, c.T_C, c.I_A, c.R_ohm)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/batteries/{battery_id}")
    def state(battery_id: str) -> Dict[str, Any]:
        try:
            return REGISTRY.state(battery_id)
        except KeyError as exc:
            raise HTTPException(404, f"unknown battery {battery_id}") from exc

    @app.get("/fleet")
    def fleet() -> List[Dict[str, Any]]:
        return REGISTRY.fleet()
except ImportError:                           # fastapi not installed: registry still usable
    app = None
