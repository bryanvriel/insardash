from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .hdf5_store import DatasetError, DatasetStore
from .schemas import SamplePointRequest, SamplePointResponse, TransectRequest, TransectResponse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"


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
    ) -> Response:
        try:
            png = store.preview_png(dataset_id, band=band, cmap=cmap, vmin=vmin, vmax=vmax, max_size=max_size)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DatasetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=30"})

    @app.post("/api/sample-point", response_model=SamplePointResponse)
    def sample_point(request: SamplePointRequest):
        try:
            samples = [
                store.sample_point(dataset_id, request.lat, request.lon, request.band)
                for dataset_id in request.dataset_ids
            ]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DatasetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"lat": request.lat, "lon": request.lon, "samples": samples}

    @app.post("/api/transect", response_model=TransectResponse)
    def transect(request: TransectRequest):
        try:
            points = [(point.lat, point.lon) for point in request.points]
            return store.transect(request.dataset_ids, request.band, points, request.samples)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (DatasetError, ValueError) as exc:
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
