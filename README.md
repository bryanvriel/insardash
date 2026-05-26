# InSAR Teaching Explorer

A LAN-hosted teaching app for exploring geocoded InSAR HDF5 files. Students can view 1-3 interferograms, adjust bands and color scales, hover for pixel values, draw shared transects, and compare profile traces.

## HDF5 Format

Each `.h5` or `.hdf5` file in `data/` represents one interferogram:

- `/data`: numeric array shaped `(N_bands, Ny, Nx)`
- `/band_names`: string array with one name per band, such as `wrapped_phase`, `unwrapped_phase`, `coherence`, `topography`
- `/lat`: latitude axis, either `(Ny,)` or `(Ny, Nx)`
- `/lon`: longitude axis, either `(Nx,)` or `(Ny, Nx)`
- optional `/units`: string array with one unit per band
- optional root attributes like `title`, `pair`, or `date`

The MVP assumes regular, monotonic latitude and longitude axes.

## Setup

Backend commands use the requested interpreter:

```bash
python3 -m pip install -r requirements.txt
```

Frontend commands require Node.js 18 or newer. Node 20 LTS is recommended:

```bash
node --version
npm --version
```

If the server has `nvm` installed:

```bash
nvm install
nvm use
```

Frontend commands use npm from that Node environment:

```bash
cd frontend
npm install
npm run build
```

Optional sample data:

```bash
python3 scripts/make_sample_data.py
```

## Run For A Class

From the repository root:

```bash
INSARDASH_DATA_DIR=insardash/data \
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Optional Tianditu satellite basemap:

```bash
INSARDASH_DATA_DIR=insardash/data \
INSARDASH_TIANDITU_KEY=YOUR_TIANDITU_KEY \
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

When `INSARDASH_TIANDITU_KEY` is set, the app exposes a Tianditu satellite background option using browser-direct tile requests. Because the browser loads tiles directly, the Tianditu key is visible to students in network traffic; use a key intended for client-side web maps and configure any domain restrictions in the Tianditu developer console.

Students on the university network can open:

```text
http://YOUR_WORKSTATION_IP:8000
```

For frontend development, run the backend on port `8000`, then:

```bash
cd frontend
npm run dev
```

The Vite dev server proxies `/api` requests to the backend.

## Tests

```bash
python3 -m pytest
cd frontend
npm run build
```
