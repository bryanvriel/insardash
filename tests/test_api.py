from __future__ import annotations

import math
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import create_app
from tests.test_hdf5_store import write_fixture


def test_api_config_without_tianditu_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("INSARDASH_TIANDITU_KEY", raising=False)
    app = create_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/config")

    assert response.status_code == 200
    assert response.json() == {
        "basemaps": [{"id": "none", "label": "None", "layers": []}],
        "default_basemap_id": "none",
    }


def test_api_config_includes_tianditu_satellite_with_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSARDASH_TIANDITU_KEY", "test-key")
    app = create_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/config")
    body = response.json()

    assert response.status_code == 200
    assert body["default_basemap_id"] == "tianditu-satellite"
    assert [basemap["id"] for basemap in body["basemaps"]] == ["none", "tianditu-satellite"]
    layer = body["basemaps"][1]["layers"][0]
    assert "T=img_w" in layer["url"]
    assert "tk=test-key" in layer["url"]
    assert layer["subdomains"] == ["0", "1", "2", "3", "4", "5", "6", "7"]


def test_api_dataset_preview_sample_and_transect(tmp_path: Path) -> None:
    write_fixture(tmp_path / "igram_a.h5")
    app = create_app(tmp_path)
    client = TestClient(app)

    listing = client.get("/api/datasets")
    assert listing.status_code == 200
    dataset_id = listing.json()[0]["id"]

    preview = client.get(f"/api/datasets/{dataset_id}/preview", params={"band": "unwrapped_phase"})
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"

    transformed_preview = client.get(
        f"/api/datasets/{dataset_id}/preview",
        params={"band": "unwrapped_phase", "transform": "np.cos(x)"},
    )
    assert transformed_preview.status_code == 200
    assert transformed_preview.content != preview.content

    invalid_preview = client.get(
        f"/api/datasets/{dataset_id}/preview",
        params={"band": "unwrapped_phase", "transform": "open('/tmp/nope')"},
    )
    assert invalid_preview.status_code == 400

    sample = client.post(
        "/api/sample-point",
        json={"dataset_ids": [dataset_id], "lat": 34.5, "lon": -117.5, "band": "unwrapped_phase"},
    )
    assert sample.status_code == 200
    assert sample.json()["samples"][0]["in_bounds"] is True
    assert "coherence" in sample.json()["samples"][0]["values"]

    fast_sample = client.post(
        "/api/sample-point",
        json={
            "dataset_ids": [dataset_id],
            "lat": 34.5,
            "lon": -117.5,
            "band": "unwrapped_phase",
            "include_all_values": False,
        },
    )
    fast_sample_body = fast_sample.json()["samples"][0]
    assert fast_sample.status_code == 200
    assert fast_sample_body["in_bounds"] is True
    assert fast_sample_body["active_band"] == "unwrapped_phase"
    assert fast_sample_body["active_value"] is not None
    assert set(fast_sample_body["values"]) == {"unwrapped_phase"}

    transformed_sample = client.post(
        "/api/sample-point",
        json={
            "maps": [{"dataset_id": dataset_id, "transform": "np.cos(x)"}],
            "lat": 34.5,
            "lon": -117.5,
            "band": "unwrapped_phase",
            "include_all_values": False,
        },
    )
    transformed_sample_body = transformed_sample.json()["samples"][0]
    assert transformed_sample.status_code == 200
    assert transformed_sample_body["transform"] == "np.cos(x)"
    assert math.isclose(transformed_sample_body["active_value"], math.cos(fast_sample_body["active_value"]))

    mixed_band_sample = client.post(
        "/api/sample-point",
        json={
            "maps": [
                {"dataset_id": dataset_id, "band": "unwrapped_phase"},
                {"dataset_id": dataset_id, "band": "coherence"},
            ],
            "lat": 34.5,
            "lon": -117.5,
            "include_all_values": False,
        },
    )
    mixed_band_sample_body = mixed_band_sample.json()["samples"]
    assert mixed_band_sample.status_code == 200
    assert [sample["active_band"] for sample in mixed_band_sample_body] == ["unwrapped_phase", "coherence"]
    assert set(mixed_band_sample_body[0]["values"]) == {"unwrapped_phase"}
    assert set(mixed_band_sample_body[1]["values"]) == {"coherence"}
    assert mixed_band_sample_body[0]["active_value"] != mixed_band_sample_body[1]["active_value"]

    transect = client.post(
        "/api/transect",
        json={
            "dataset_ids": [dataset_id],
            "band": "unwrapped_phase",
            "points": [{"lat": 34.95, "lon": -117.95}, {"lat": 34.05, "lon": -117.05}],
            "samples": 24,
        },
    )
    assert transect.status_code == 200
    assert len(transect.json()["profiles"][0]["values"]) == 24

    transformed_transect = client.post(
        "/api/transect",
        json={
            "maps": [{"dataset_id": dataset_id, "transform": "x + 1"}],
            "band": "unwrapped_phase",
            "points": [{"lat": 34.95, "lon": -117.95}, {"lat": 34.05, "lon": -117.05}],
            "samples": 24,
        },
    )
    assert transformed_transect.status_code == 200
    raw_profile = transect.json()["profiles"][0]["values"]
    transformed_profile = transformed_transect.json()["profiles"][0]
    assert transformed_profile["transform"] == "x + 1"
    assert math.isclose(transformed_profile["values"][0], raw_profile[0] + 1, abs_tol=1e-6)

    mixed_band_transect = client.post(
        "/api/transect",
        json={
            "maps": [
                {"dataset_id": dataset_id, "band": "unwrapped_phase"},
                {"dataset_id": dataset_id, "band": "coherence", "transform": "np.abs(x)"},
            ],
            "points": [{"lat": 34.95, "lon": -117.95}, {"lat": 34.05, "lon": -117.05}],
            "samples": 24,
        },
    )
    mixed_band_transect_body = mixed_band_transect.json()
    assert mixed_band_transect.status_code == 200
    assert mixed_band_transect_body["band"] == "mixed"
    assert [profile["band"] for profile in mixed_band_transect_body["profiles"]] == ["unwrapped_phase", "coherence"]
    assert mixed_band_transect_body["profiles"][0]["units"] == "rad"
    assert mixed_band_transect_body["profiles"][1]["units"] == "1"
    assert mixed_band_transect_body["profiles"][1]["transform"] == "np.abs(x)"

    missing_band_sample = client.post(
        "/api/sample-point",
        json={"maps": [{"dataset_id": dataset_id}], "lat": 34.5, "lon": -117.5, "include_all_values": False},
    )
    assert missing_band_sample.status_code == 422

    missing_band_transect = client.post(
        "/api/transect",
        json={
            "maps": [{"dataset_id": dataset_id}],
            "points": [{"lat": 34.95, "lon": -117.95}, {"lat": 34.05, "lon": -117.05}],
            "samples": 24,
        },
    )
    assert missing_band_transect.status_code == 422
