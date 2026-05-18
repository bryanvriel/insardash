from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Bounds(BaseModel):
    south: float
    west: float
    north: float
    east: float


class RasterShape(BaseModel):
    bands: int
    rows: int
    cols: int


class BandInfo(BaseModel):
    name: str
    index: int
    units: str | None = None


class DatasetSummary(BaseModel):
    id: str
    filename: str
    title: str
    shape: RasterShape
    bounds: Bounds
    bands: list[BandInfo]
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeoPoint(BaseModel):
    lat: float
    lon: float


class SamplePointRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=3)
    lat: float
    lon: float
    band: str | None = None
    include_all_values: bool = True


class DatasetPointSample(BaseModel):
    dataset_id: str
    title: str
    in_bounds: bool
    row: int | None = None
    col: int | None = None
    active_band: str | None = None
    active_value: float | None = None
    values: dict[str, float | None] = Field(default_factory=dict)
    units: dict[str, str | None] = Field(default_factory=dict)


class SamplePointResponse(BaseModel):
    lat: float
    lon: float
    samples: list[DatasetPointSample]


class TransectRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=3)
    band: str
    points: list[GeoPoint] = Field(min_length=2)
    samples: int = Field(default=256, ge=2, le=4096)


class TransectProfile(BaseModel):
    dataset_id: str
    title: str
    band: str
    units: str | None = None
    values: list[float | None]


class TransectResponse(BaseModel):
    band: str
    distance_km: list[float]
    lat: list[float]
    lon: list[float]
    profiles: list[TransectProfile]
