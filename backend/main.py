from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .hdf5_store import DatasetError, DatasetStore
from .schemas import AppConfig, Basemap, BasemapLayer, MapSelection, SamplePointRequest, SamplePointResponse, TransectRequest, TransectResponse
from .value_transform import TransformError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
TIANDITU_KEY_ENV = "INSARDASH_TIANDITU_KEY"
TIANDITU_SUBDOMAINS = [str(index) for index in range(8)]


def _request_maps(dataset_ids: list[str] | None, maps: list[MapSelection] | None) -> list[MapSelection]:
    if maps is not None:
        return maps
    if dataset_ids is not None:
        return [MapSelection(dataset_id=dataset_id) for dataset_id in dataset_ids]
    raise HTTPException(status_code=422, detail="Request must include maps or dataset_ids")


def _resolve_band(map_selection: MapSelection, fallback_band: str | None) -> str:
    band = map_selection.band or fallback_band
    if band is None:
        raise HTTPException(status_code=422, detail="Each map requires a band or a request-level band")
    return band


def _app_config() -> AppConfig:
    basemaps = [Basemap(id="none", label="None", layers=[])]
    tianditu_key = os.getenv(TIANDITU_KEY_ENV, "").strip()
    if tianditu_key:
        basemaps.append(
            Basemap(
                id="tianditu-satellite",
                label="Tianditu Satellite",
                layers=[
                    BasemapLayer(
                        url=f"https://t{{s}}.tianditu.gov.cn/DataServer?T=img_w&x={{x}}&y={{y}}&l={{z}}&tk={quote(tianditu_key, safe='')}",
                        subdomains=TIANDITU_SUBDOMAINS,
                        attribution='&copy; <a href="https://www.tianditu.gov.cn/">Tianditu</a>',
                        max_zoom=18,
                    )
                ],
            )
        )
    return AppConfig(basemaps=basemaps, default_basemap_id=basemaps[-1].id)


def create_app(data_dir: Path | None = None) -> FastAPI:
    store = DatasetStore(Path(data_dir or os.getenv("INSARDASH_DATA_DIR", DEFAULT_DATA_DIR)))
    app = FastAPI(title="InSAR Teaching Explorer", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config", response_model=AppConfig)
    def config() -> AppConfig:
        return _app_config()

    @app.get("/api/datasets")
    def datasets():
        try:
            return store.list_datasets()
        except DatasetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/preview")
    def preview(
        dataset_id: str,
        band: str,
        cmap: str = Query(default="viridis"),
        vmin: float | None = Query(default=None),
        vmax: float | None = Query(default=None),
        max_size: int = Query(default=1200, ge=64, le=4096),
        transform: str | None = Query(default=None),
    ) -> Response:
        try:
            png = store.preview_png(dataset_id, band=band, cmap=cmap, vmin=vmin, vmax=vmax, max_size=max_size, transform=transform)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TransformError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DatasetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=30"})

    @app.post("/api/sample-point", response_model=SamplePointResponse)
    def sample_point(request: SamplePointRequest):
        try:
            maps = _request_maps(request.dataset_ids, request.maps)
            samples = [
                store.sample_point(
                    map_selection.dataset_id,
                    request.lat,
                    request.lon,
                    _resolve_band(map_selection, request.band),
                    include_all_values=request.include_all_values,
                    transform=map_selection.transform,
                )
                for map_selection in maps
            ]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (DatasetError, TransformError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"lat": request.lat, "lon": request.lon, "samples": samples}

    @app.post("/api/transect", response_model=TransectResponse)
    def transect(request: TransectRequest):
        try:
            maps = _request_maps(request.dataset_ids, request.maps)
            points = [(point.lat, point.lon) for point in request.points]
            return store.transect(
                [map_selection.dataset_id for map_selection in maps],
                request.band,
                points,
                request.samples,
                transforms=[map_selection.transform for map_selection in maps],
                bands=[_resolve_band(map_selection, request.band) for map_selection in maps],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (DatasetError, TransformError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    frontend_dist = ROOT / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            file_path = frontend_dist / path
            if path and file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
