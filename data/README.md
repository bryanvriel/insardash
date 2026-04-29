# Data Folder

Place classroom HDF5 interferograms here. Each file should include:

- `/data`: a numeric array shaped `(N_bands, Ny, Nx)`
- `/band_names`: one string per band
- `/lat`: latitude axis, either `(Ny,)` or `(Ny, Nx)`
- `/lon`: longitude axis, either `(Nx,)` or `(Ny, Nx)`
- optional `/units`: one string per band

The backend scans this folder for `.h5` and `.hdf5` files.
