import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";
import {
  Check,
  Layers,
  LocateFixed,
  MousePointer2,
  Play,
  RotateCcw,
  Spline,
  Trash2,
  X
} from "lucide-react";
import type { LatLngBoundsExpression } from "leaflet";
import { CircleMarker, ImageOverlay, MapContainer, Polyline, ScaleControl, useMap, useMapEvents } from "react-leaflet";
import Plot from "react-plotly.js";

type Bounds = {
  south: number;
  west: number;
  north: number;
  east: number;
};

type BandInfo = {
  name: string;
  index: number;
  units: string | null;
};

type DatasetSummary = {
  id: string;
  filename: string;
  title: string;
  shape: {
    bands: number;
    rows: number;
    cols: number;
  };
  bounds: Bounds;
  bands: BandInfo[];
  metadata: Record<string, string | number | boolean>;
};

type GeoPoint = {
  lat: number;
  lon: number;
};

type Viewport = {
  center: [number, number];
  zoom: number;
};

type Transect = {
  id: string;
  name: string;
  color: string;
  points: GeoPoint[];
};

type MapColorSettings = {
  band: string;
  cmap: string;
  autoScale: boolean;
  vmin: string;
  vmax: string;
  transform: string;
};

type SelectedMapSlot = {
  slotIndex: number;
  dataset: DatasetSummary;
};

type MapRequest = {
  dataset_id: string;
  band: string;
  transform?: string;
};

type PointSample = {
  dataset_id: string;
  title: string;
  in_bounds: boolean;
  row?: number | null;
  col?: number | null;
  active_band?: string | null;
  active_value?: number | null;
  transform?: string | null;
  values: Record<string, number | null>;
  units: Record<string, string | null>;
};

type HoverInfo = {
  lat: number;
  lon: number;
  samples: PointSample[];
};

type TransectResponse = {
  band: string;
  distance_km: number[];
  lat: number[];
  lon: number[];
  profiles: Array<{
    dataset_id: string;
    title: string;
    band: string;
    units: string | null;
    transform?: string | null;
    values: Array<number | null>;
  }>;
};

const COLORMAPS = ["viridis", "plasma", "inferno", "magma", "turbo", "gray", "twilight", "RdBu_r"];
const MAP_SLOTS = [0, 1, 2];
const HOVER_THROTTLE_MS = 50;
const DEFAULT_PROFILE_HEIGHT_PX = 288;
const MIN_PROFILE_HEIGHT_PX = 192;
const MIN_MAP_ROW_HEIGHT_PX = 288;
const MAX_PROFILE_HEIGHT_RATIO = 0.55;
const RESIZE_STEP_PX = 16;
const RESIZE_LARGE_STEP_PX = 48;
const TWO_COLUMN_LAYOUT_MEDIA = "(max-width: 1120px)";
const STACKED_LAYOUT_MEDIA = "(max-width: 780px)";
const TRANSECT_COLORS = ["#d94f35", "#0b8f7a", "#6d5dfc", "#c47b00", "#0077b6", "#b2386b"];
const PlotComponent = Plot as any;

function defaultMapColorSettings(): MapColorSettings {
  return {
    band: "",
    cmap: "viridis",
    autoScale: true,
    vmin: "",
    vmax: "",
    transform: ""
  };
}

function formatValue(value: number | null | undefined, units?: string | null) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "No data";
  }
  const abs = Math.abs(value);
  const formatted = abs >= 1000 || abs < 0.01 ? value.toExponential(3) : value.toFixed(3);
  return units ? `${formatted} ${units}` : formatted;
}

function formatCoord(value: number) {
  return value.toFixed(5);
}

function boundsExpression(bounds: Bounds): LatLngBoundsExpression {
  return [
    [bounds.south, bounds.west],
    [bounds.north, bounds.east]
  ];
}

function centerOfBounds(bounds: Bounds): [number, number] {
  return [(bounds.south + bounds.north) / 2, (bounds.west + bounds.east) / 2];
}

function normalizeTransform(transform: string) {
  return transform.trim();
}

function makePreviewUrl(datasetId: string, band: string, cmap: string, autoScale: boolean, vmin: string, vmax: string, transform: string) {
  const params = new URLSearchParams({ band, cmap, max_size: "1400" });
  if (!autoScale && vmin.trim() !== "") {
    params.set("vmin", vmin.trim());
  }
  if (!autoScale && vmax.trim() !== "") {
    params.set("vmax", vmax.trim());
  }
  const normalizedTransform = normalizeTransform(transform);
  if (normalizedTransform) {
    params.set("transform", normalizedTransform);
  }
  return `/api/datasets/${encodeURIComponent(datasetId)}/preview?${params.toString()}`;
}

function stableId() {
  if ("crypto" in window && "randomUUID" in window.crypto) {
    return window.crypto.randomUUID();
  }
  return `transect-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function SyncMap({
  viewport,
  bounds,
  drawing,
  onViewportChange,
  onHover,
  onAddDraftPoint
}: {
  viewport: Viewport | null;
  bounds: Bounds;
  drawing: boolean;
  onViewportChange: (viewport: Viewport) => void;
  onHover: (point: GeoPoint) => void;
  onAddDraftPoint: (point: GeoPoint) => void;
}) {
  const map = useMap();
  const internalMove = useRef(false);
  const fitKey = `${bounds.south}:${bounds.west}:${bounds.north}:${bounds.east}`;

  useEffect(() => {
    if (viewport) {
      const current = map.getCenter();
      if (
        Math.abs(current.lat - viewport.center[0]) > 1e-7 ||
        Math.abs(current.lng - viewport.center[1]) > 1e-7 ||
        map.getZoom() !== viewport.zoom
      ) {
        internalMove.current = true;
        map.setView(viewport.center, viewport.zoom, { animate: false });
        window.setTimeout(() => {
          internalMove.current = false;
        }, 0);
      }
      return;
    }
    map.fitBounds(boundsExpression(bounds), { padding: [14, 14], animate: false });
  }, [bounds, fitKey, map, viewport]);

  useMapEvents({
    moveend() {
      if (internalMove.current) {
        return;
      }
      const center = map.getCenter();
      onViewportChange({ center: [center.lat, center.lng], zoom: map.getZoom() });
    },
    mousemove(event) {
      onHover({ lat: event.latlng.lat, lon: event.latlng.lng });
    },
    click(event) {
      if (drawing) {
        onAddDraftPoint({ lat: event.latlng.lat, lon: event.latlng.lng });
      }
    }
  });

  return null;
}

function MapPanel({
  dataset,
  band,
  cmap,
  autoScale,
  vmin,
  vmax,
  transform,
  viewport,
  drawing,
  transects,
  draftPoints,
  onViewportChange,
  onHover,
  onAddDraftPoint
}: {
  dataset: DatasetSummary;
  band: string;
  cmap: string;
  autoScale: boolean;
  vmin: string;
  vmax: string;
  transform: string;
  viewport: Viewport | null;
  drawing: boolean;
  transects: Transect[];
  draftPoints: GeoPoint[];
  onViewportChange: (viewport: Viewport) => void;
  onHover: (point: GeoPoint) => void;
  onAddDraftPoint: (point: GeoPoint) => void;
}) {
  const previewUrl = useMemo(
    () => makePreviewUrl(dataset.id, band, cmap, autoScale, vmin, vmax, transform),
    [autoScale, band, cmap, dataset.id, transform, vmax, vmin]
  );
  const mapBounds = boundsExpression(dataset.bounds);
  const initialCenter = centerOfBounds(dataset.bounds);

  return (
    <section className="map-panel" aria-label={`${dataset.title} map`}>
      <div className="map-titlebar">
        <div>
          <strong>{dataset.title}</strong>
          <span>{dataset.shape.rows.toLocaleString()} x {dataset.shape.cols.toLocaleString()}</span>
        </div>
        <span className="map-band">{band}</span>
      </div>
      <MapContainer
        className="map"
        center={initialCenter}
        zoom={9}
        minZoom={1}
        maxZoom={18}
        doubleClickZoom={false}
        scrollWheelZoom
        attributionControl={false}
      >
        <ImageOverlay url={previewUrl} bounds={mapBounds} opacity={1} />
        {transects.map((transect) => (
          <Polyline
            key={transect.id}
            pathOptions={{ color: transect.color, weight: 3, opacity: 0.95 }}
            positions={transect.points.map((point) => [point.lat, point.lon])}
          />
        ))}
        {draftPoints.length > 0 && (
          <>
            <Polyline
              pathOptions={{ color: "#101828", weight: 3, dashArray: "6 6" }}
              positions={draftPoints.map((point) => [point.lat, point.lon])}
            />
            {draftPoints.map((point, index) => (
              <CircleMarker
                key={`${point.lat}:${point.lon}:${index}`}
                center={[point.lat, point.lon]}
                radius={5}
                pathOptions={{ color: "#101828", fillColor: "#ffffff", fillOpacity: 1, weight: 2 }}
              />
            ))}
          </>
        )}
        <ScaleControl position="bottomleft" metric imperial={false} />
        <SyncMap
          viewport={viewport}
          bounds={dataset.bounds}
          drawing={drawing}
          onViewportChange={onViewportChange}
          onHover={onHover}
          onAddDraftPoint={onAddDraftPoint}
        />
      </MapContainer>
    </section>
  );
}

function ProfilePlot({
  transects,
  responses
}: {
  transects: Transect[];
  responses: Record<string, TransectResponse>;
}) {
  const traces = transects.flatMap((transect) => {
    const response = responses[transect.id];
    if (!response) {
      return [];
    }
    return response.profiles.map((profile, index) => {
      const bandLabel = profile.units ? `${profile.band} (${profile.units})` : profile.band;
      const transformLabel = profile.transform ? ` (${profile.transform})` : "";
      return {
        x: response.distance_km,
        y: profile.values,
        mode: "lines",
        type: "scatter",
        name: `${transect.name} - ${profile.title} - ${bandLabel}${transformLabel}`,
        line: {
          color: transect.color,
          width: index === 0 ? 3 : 2,
          dash: index === 0 ? "solid" : index === 1 ? "dash" : "dot"
        },
        hovertemplate: `Distance %{x:.2f} km<br>${bandLabel}: %{y:.3f}<extra>%{fullData.name}</extra>`
      };
    });
  });

  if (traces.length === 0) {
    return (
      <div className="empty-plot">
        <Spline size={26} />
        <span>Draw a transect to compare profiles.</span>
      </div>
    );
  }

  return (
    <PlotComponent
      data={traces}
      layout={{
        autosize: true,
        margin: { l: 64, r: 20, t: 18, b: 52 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#ffffff",
        xaxis: { title: "Distance (km)", zeroline: false, gridcolor: "#e6e8ee" },
        yaxis: { title: "Value", zeroline: false, gridcolor: "#e6e8ee" },
        legend: { orientation: "h", y: -0.26 },
        font: { family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 12, color: "#1f2937" }
      }}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
      className="plot"
    />
  );
}

export default function App() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<Array<string | null>>([null, null, null]);
  const [mapColorSettings, setMapColorSettings] = useState<MapColorSettings[]>(() => MAP_SLOTS.map(() => defaultMapColorSettings()));
  const [profileHeightPx, setProfileHeightPx] = useState(DEFAULT_PROFILE_HEIGHT_PX);
  const [isProfileResizing, setIsProfileResizing] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [draftPoints, setDraftPoints] = useState<GeoPoint[]>([]);
  const [transects, setTransects] = useState<Transect[]>([]);
  const [profileResponses, setProfileResponses] = useState<Record<string, TransectResponse>>({});
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [viewport, setViewport] = useState<Viewport | null>(null);
  const [status, setStatus] = useState("Loading datasets...");
  const [profileStatus, setProfileStatus] = useState("");
  const workspaceRef = useRef<HTMLElement | null>(null);
  const hoverTimer = useRef<number | null>(null);
  const hoverAbort = useRef<AbortController | null>(null);
  const hoverRequestId = useRef(0);
  const pendingHoverPoint = useRef<GeoPoint | null>(null);
  const lastHoverRequestAt = useRef(0);
  const profileResize = useRef<{ pointerId: number; startY: number; startHeight: number } | null>(null);

  const getProfileHeightBounds = useCallback(() => {
    const workspace = workspaceRef.current;
    const workspaceHeight = workspace?.getBoundingClientRect().height ?? window.innerHeight;
    const rowGap = workspace ? Number.parseFloat(window.getComputedStyle(workspace).rowGap || "0") || 0 : 12;
    const readoutPanel = workspace?.querySelector<HTMLElement>(".readout-panel") ?? null;
    const readoutRowHeight =
      window.matchMedia(TWO_COLUMN_LAYOUT_MEDIA).matches && !window.matchMedia(STACKED_LAYOUT_MEDIA).matches
        ? readoutPanel?.getBoundingClientRect().height ?? 0
        : 0;
    const rowGapCount = readoutRowHeight > 0 ? 2 : 1;
    const maxByRatio = workspaceHeight * MAX_PROFILE_HEIGHT_RATIO;
    const maxByMap = workspaceHeight - MIN_MAP_ROW_HEIGHT_PX - readoutRowHeight - rowGap * rowGapCount;
    const max = Math.max(MIN_PROFILE_HEIGHT_PX, Math.min(maxByRatio, maxByMap));
    return { min: MIN_PROFILE_HEIGHT_PX, max: Math.round(max) };
  }, []);

  const clampProfileHeight = useCallback(
    (height: number) => {
      const { min, max } = getProfileHeightBounds();
      return Math.round(Math.min(Math.max(height, min), max));
    },
    [getProfileHeightBounds]
  );

  const profileHeightBounds = getProfileHeightBounds();
  const workspaceStyle: CSSProperties & { "--profile-height": string } = {
    "--profile-height": `${profileHeightPx}px`
  };

  useEffect(() => {
    let cancelled = false;
    fetch("/api/datasets")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Dataset request failed with ${response.status}`);
        }
        return response.json() as Promise<DatasetSummary[]>;
      })
      .then((items) => {
        if (cancelled) {
          return;
        }
        setDatasets(items);
        const defaults = items.slice(0, Math.min(items.length, 3)).map((item) => item.id);
        setSelectedIds([defaults[0] ?? null, defaults[1] ?? null, defaults[2] ?? null]);
        setStatus(items.length ? `${items.length} dataset${items.length === 1 ? "" : "s"} ready.` : "No HDF5 files found in the data folder.");
      })
      .catch((error) => {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Unable to load datasets.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedMapSlots = useMemo(
    () =>
      selectedIds
        .map((id, slotIndex) => {
          const dataset = datasets.find((item) => item.id === id);
          return dataset ? { slotIndex, dataset } : null;
        })
        .filter((slot): slot is SelectedMapSlot => Boolean(slot)),
    [datasets, selectedIds]
  );

  const selectedDatasets = useMemo(() => selectedMapSlots.map((slot) => slot.dataset), [selectedMapSlots]);
  useEffect(() => {
    setMapColorSettings((items) =>
      items.map((settings, index) => {
        const dataset = datasets.find((item) => item.id === selectedIds[index]);
        if (!dataset) {
          return settings.band ? { ...settings, band: "" } : settings;
        }
        if (dataset.bands.some((band) => band.name === settings.band)) {
          return settings;
        }
        return { ...settings, band: dataset.bands[0]?.name ?? "" };
      })
    );
  }, [datasets, selectedIds]);

  const mapBandKey = mapColorSettings.map((settings) => settings.band).join("|");
  const mapTransformKey = mapColorSettings.map((settings) => normalizeTransform(settings.transform)).join("|");
  const selectedMapRequests = useMemo<MapRequest[]>(
    () =>
      selectedMapSlots.map(({ dataset, slotIndex }) => {
        const band = mapColorSettings[slotIndex].band;
        const transform = normalizeTransform(mapColorSettings[slotIndex].transform);
        return transform ? { dataset_id: dataset.id, band, transform } : { dataset_id: dataset.id, band };
      }),
    [mapBandKey, mapTransformKey, selectedMapSlots]
  );
  const selectedMapKey = selectedMapRequests
    .map((request) => `${request.dataset_id}:${request.band}:${request.transform ?? ""}`)
    .join("|");
  const selectedMapsHaveBands = selectedMapRequests.every((request) => Boolean(request.band));

  const flushHover = useCallback(() => {
    hoverTimer.current = null;
    const point = pendingHoverPoint.current;
    pendingHoverPoint.current = null;
    if (!point || selectedMapRequests.length === 0 || !selectedMapsHaveBands) {
      return;
    }

    hoverAbort.current?.abort();
    const controller = new AbortController();
    const requestId = hoverRequestId.current + 1;
    hoverRequestId.current = requestId;
    hoverAbort.current = controller;
    lastHoverRequestAt.current = window.performance.now();
    fetch("/api/sample-point", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        maps: selectedMapRequests,
        lat: point.lat,
        lon: point.lon,
        include_all_values: false
      }),
      signal: controller.signal
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Hover sample failed with ${response.status}`);
        }
        return response.json() as Promise<HoverInfo>;
      })
      .then((sample) => {
        if (requestId === hoverRequestId.current) {
          setHoverInfo(sample);
        }
      })
      .catch((error) => {
        if (requestId === hoverRequestId.current && !(error instanceof DOMException && error.name === "AbortError")) {
          setHoverInfo({ lat: point.lat, lon: point.lon, samples: [] });
        }
      });
  }, [selectedMapRequests, selectedMapsHaveBands]);

  const handleHover = useCallback(
    (point: GeoPoint) => {
      pendingHoverPoint.current = point;
      if (hoverTimer.current !== null) {
        return;
      }
      const elapsed = window.performance.now() - lastHoverRequestAt.current;
      const wait = Math.max(0, HOVER_THROTTLE_MS - elapsed);
      hoverTimer.current = window.setTimeout(flushHover, wait);
    },
    [flushHover]
  );

  useEffect(() => {
    hoverRequestId.current += 1;
    pendingHoverPoint.current = null;
    if (hoverTimer.current !== null) {
      window.clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
    }
    hoverAbort.current?.abort();
    hoverAbort.current = null;
  }, [selectedMapKey]);

  useEffect(() => {
    return () => {
      if (hoverTimer.current !== null) {
        window.clearTimeout(hoverTimer.current);
      }
      hoverAbort.current?.abort();
    };
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setProfileHeightPx((height) => clampProfileHeight(height));
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [clampProfileHeight]);

  useEffect(() => {
    const animationFrame = window.requestAnimationFrame(() => {
      window.dispatchEvent(new Event("resize"));
    });
    return () => window.cancelAnimationFrame(animationFrame);
  }, [profileHeightPx]);

  useEffect(() => {
    if (transects.length === 0 || selectedMapRequests.length === 0 || !selectedMapsHaveBands) {
      setProfileResponses({});
      return;
    }
    const controller = new AbortController();
    setProfileStatus("Updating profiles...");
    Promise.all(
      transects.map((transect) =>
        fetch("/api/transect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            maps: selectedMapRequests,
            points: transect.points,
            samples: 256
          }),
          signal: controller.signal
        }).then((response) => {
          if (!response.ok) {
            throw new Error(`Transect request failed with ${response.status}`);
          }
          return response.json() as Promise<TransectResponse>;
        })
      )
    )
      .then((responses) => {
        const next: Record<string, TransectResponse> = {};
        responses.forEach((response, index) => {
          next[transects[index].id] = response;
        });
        setProfileResponses(next);
        setProfileStatus("");
      })
      .catch((error) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setProfileStatus(error instanceof Error ? error.message : "Unable to update profiles.");
        }
      });
    return () => controller.abort();
  }, [selectedMapKey, selectedMapRequests, selectedMapsHaveBands, transects]);

  function updateSelectedId(index: number, value: string) {
    const next = [...selectedIds];
    next[index] = value || null;
    setSelectedIds(next);
    setViewport(null);
  }

  function updateMapColorSettings(index: number, changes: Partial<MapColorSettings>) {
    setMapColorSettings((items) =>
      items.map((settings, itemIndex) =>
        itemIndex === index ? { ...settings, ...changes } : settings
      )
    );
  }

  function startProfileResize(event: ReactPointerEvent<HTMLDivElement>) {
    if (window.matchMedia(STACKED_LAYOUT_MEDIA).matches) {
      return;
    }
    event.preventDefault();
    profileResize.current = {
      pointerId: event.pointerId,
      startY: event.clientY,
      startHeight: profileHeightPx
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsProfileResizing(true);
  }

  function updateProfileResize(event: ReactPointerEvent<HTMLDivElement>) {
    const resize = profileResize.current;
    if (!resize || resize.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const nextHeight = resize.startHeight - (event.clientY - resize.startY);
    setProfileHeightPx(clampProfileHeight(nextHeight));
  }

  function stopProfileResize(event: ReactPointerEvent<HTMLDivElement>) {
    const resize = profileResize.current;
    if (!resize || resize.pointerId !== event.pointerId) {
      return;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    profileResize.current = null;
    setIsProfileResizing(false);
  }

  function handleProfileResizeKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    let nextHeight: number | null = null;
    if (event.key === "ArrowUp") {
      nextHeight = profileHeightPx + RESIZE_STEP_PX;
    } else if (event.key === "ArrowDown") {
      nextHeight = profileHeightPx - RESIZE_STEP_PX;
    } else if (event.key === "PageUp") {
      nextHeight = profileHeightPx + RESIZE_LARGE_STEP_PX;
    } else if (event.key === "PageDown") {
      nextHeight = profileHeightPx - RESIZE_LARGE_STEP_PX;
    } else if (event.key === "Home") {
      nextHeight = profileHeightBounds.min;
    } else if (event.key === "End") {
      nextHeight = profileHeightBounds.max;
    }
    if (nextHeight === null) {
      return;
    }
    event.preventDefault();
    setProfileHeightPx(clampProfileHeight(nextHeight));
  }

  function finishDraft() {
    if (draftPoints.length < 2) {
      return;
    }
    const transect: Transect = {
      id: stableId(),
      name: `Transect ${transects.length + 1}`,
      color: TRANSECT_COLORS[transects.length % TRANSECT_COLORS.length],
      points: draftPoints
    };
    setTransects((items) => [...items, transect]);
    setDraftPoints([]);
    setDrawing(false);
  }

  function cancelDraft() {
    setDraftPoints([]);
    setDrawing(false);
  }

  function clearTransects() {
    setTransects([]);
    setDraftPoints([]);
    setProfileResponses({});
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>InSAR Teaching Explorer</h1>
          <p>{status}</p>
        </div>
        <div className="header-badge">
          <LocateFixed size={16} />
          LAN classroom mode
        </div>
      </header>

      <main
        ref={workspaceRef}
        className={isProfileResizing ? "workspace profile-resizing" : "workspace"}
        style={workspaceStyle}
      >
        <aside className="control-panel" aria-label="Display controls">
          <section className="control-section">
            <h2><Layers size={17} /> Maps</h2>
            {MAP_SLOTS.map((index) => {
              const selectedDataset = datasets.find((dataset) => dataset.id === selectedIds[index]);
              const bandOptions = selectedDataset?.bands ?? [];
              return (
                <div key={index} className="map-control-group">
                  <label className="field">
                    <span>Map {index + 1}</span>
                    <select value={selectedIds[index] ?? ""} onChange={(event) => updateSelectedId(index, event.target.value)}>
                      <option value="">None</option>
                      {datasets.map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>
                          {dataset.title}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Band</span>
                    <select
                      value={mapColorSettings[index].band}
                      onChange={(event) => updateMapColorSettings(index, { band: event.target.value })}
                      disabled={!selectedDataset || bandOptions.length === 0}
                    >
                      {bandOptions.map((band) => (
                        <option key={band.name} value={band.name}>
                          {band.name}{band.units ? ` (${band.units})` : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Colormap</span>
                    <select
                      value={mapColorSettings[index].cmap}
                      onChange={(event) => updateMapColorSettings(index, { cmap: event.target.value })}
                    >
                      {COLORMAPS.map((name) => (
                        <option key={name} value={name}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Transform</span>
                    <input
                      value={mapColorSettings[index].transform}
                      onChange={(event) => updateMapColorSettings(index, { transform: event.target.value })}
                      placeholder="np.abs(x)"
                      spellCheck={false}
                    />
                  </label>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={mapColorSettings[index].autoScale}
                      onChange={(event) => updateMapColorSettings(index, { autoScale: event.target.checked })}
                    />
                    <span>Auto scale</span>
                  </label>
                  <div className="scale-row">
                    <label className="field compact">
                      <span>Min</span>
                      <input
                        value={mapColorSettings[index].vmin}
                        onChange={(event) => updateMapColorSettings(index, { vmin: event.target.value })}
                        disabled={mapColorSettings[index].autoScale}
                        inputMode="decimal"
                      />
                    </label>
                    <label className="field compact">
                      <span>Max</span>
                      <input
                        value={mapColorSettings[index].vmax}
                        onChange={(event) => updateMapColorSettings(index, { vmax: event.target.value })}
                        disabled={mapColorSettings[index].autoScale}
                        inputMode="decimal"
                      />
                    </label>
                  </div>
                </div>
              );
            })}
          </section>

          <section className="control-section">
            <h2><Spline size={17} /> Transects</h2>
            <div className="button-row">
              <button
                type="button"
                className={drawing ? "primary active" : "primary"}
                onClick={() => setDrawing(true)}
                disabled={selectedDatasets.length === 0}
                title="Draw transect"
              >
                <MousePointer2 size={16} />
                Draw
              </button>
              <button type="button" onClick={finishDraft} disabled={draftPoints.length < 2} title="Finish transect">
                <Check size={16} />
                Finish
              </button>
              <button type="button" onClick={cancelDraft} disabled={!drawing && draftPoints.length === 0} title="Cancel drawing">
                <X size={16} />
              </button>
            </div>
            <div className="button-row">
              <button type="button" onClick={clearTransects} disabled={transects.length === 0 && draftPoints.length === 0} title="Clear transects">
                <Trash2 size={16} />
                Clear
              </button>
              <button type="button" onClick={() => setViewport(null)} title="Refit maps">
                <RotateCcw size={16} />
                Refit
              </button>
            </div>
            <div className="transect-list">
              {transects.length === 0 && <span className="muted">No transects yet.</span>}
              {transects.map((transect) => (
                <div key={transect.id} className="transect-item">
                  <span style={{ backgroundColor: transect.color }} />
                  <strong>{transect.name}</strong>
                  <button
                    type="button"
                    title={`Delete ${transect.name}`}
                    onClick={() => setTransects((items) => items.filter((item) => item.id !== transect.id))}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
            {drawing && <p className="hint">Click points on any map, then finish the transect.</p>}
          </section>
        </aside>

        <section className="map-grid" aria-label="Linked interferogram maps">
          {selectedDatasets.length === 0 ? (
            <div className="empty-state">
              <Layers size={28} />
              <span>Add HDF5 files to the data folder, then reload the page.</span>
            </div>
          ) : !selectedMapsHaveBands ? (
            <div className="empty-state">
              <Layers size={28} />
              <span>Select a band for each map.</span>
            </div>
          ) : (
            selectedMapSlots.map(({ dataset, slotIndex }) => {
              const colorSettings = mapColorSettings[slotIndex];
              return (
                <MapPanel
                  key={`${dataset.id}-${slotIndex}`}
                  dataset={dataset}
                  band={colorSettings.band}
                  cmap={colorSettings.cmap}
                  autoScale={colorSettings.autoScale}
                  vmin={colorSettings.vmin}
                  vmax={colorSettings.vmax}
                  transform={colorSettings.transform}
                  viewport={viewport}
                  drawing={drawing}
                  transects={transects}
                  draftPoints={draftPoints}
                  onViewportChange={setViewport}
                  onHover={handleHover}
                  onAddDraftPoint={(point) => setDraftPoints((items) => [...items, point])}
                />
              );
            })
          )}
        </section>

        <section className="readout-panel" aria-label="Hover readout">
          <h2><Play size={17} /> Hover Values</h2>
          {hoverInfo ? (
            <>
              <div className="coord-row">
                <span>{formatCoord(hoverInfo.lat)}</span>
                <span>{formatCoord(hoverInfo.lon)}</span>
              </div>
              <div className="sample-list">
                {hoverInfo.samples.map((sample, index) => (
                  <div key={`${sample.dataset_id}:${index}`} className="sample-row">
                    <strong>
                      {sample.title}{sample.active_band ? ` - ${sample.active_band}` : ""}{sample.transform ? ` (${sample.transform})` : ""}
                    </strong>
                    <span>{sample.in_bounds ? formatValue(sample.active_value, sample.units[sample.active_band ?? ""]) : "Outside raster"}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="muted">Move over a map to inspect values.</p>
          )}
        </section>

        <section className="profile-panel" aria-label="Transect profile plot">
          <div
            className="profile-resize-handle"
            role="separator"
            aria-label="Resize profiles panel"
            aria-orientation="horizontal"
            aria-valuemin={profileHeightBounds.min}
            aria-valuemax={profileHeightBounds.max}
            aria-valuenow={profileHeightPx}
            tabIndex={0}
            title="Drag to resize profiles"
            onPointerDown={startProfileResize}
            onPointerMove={updateProfileResize}
            onPointerUp={stopProfileResize}
            onPointerCancel={stopProfileResize}
            onLostPointerCapture={() => {
              profileResize.current = null;
              setIsProfileResizing(false);
            }}
            onKeyDown={handleProfileResizeKeyDown}
          />
          <div className="profile-heading">
            <h2>Profiles</h2>
            <span>{profileStatus || `${transects.length} transect${transects.length === 1 ? "" : "s"}`}</span>
          </div>
          <ProfilePlot transects={transects} responses={profileResponses} />
        </section>
      </main>
    </div>
  );
}
