# RSPM Command Center

Production-ready Streamlit dashboard for Reservoir and Surface Pipeline Monitoring (RSPM). The app combines engineering calculations, ML-based leak-risk classification, scenario uploads, and report export in a single interface.

## Features

- Modern Streamlit dashboard for reservoir and pipeline monitoring
- Engineering metrics including failure pressure, integrity margin, and failure probability
- ML-based risk diagnosis using the trained model in `models/rspm_models.pkl`
- Scenario selection from the integrated dataset
- CSV upload support with validation and template download
- Categorised reservoir, pipeline, and fluid-chemistry analysis
- Excel and PDF report export

## Project Structure

- `streamlit_app.py`: top-level hosting entrypoint
- `dashboard/app.py`: main Streamlit dashboard
- `data/rspm_integrated.csv`: integrated scenario dataset
- `data/RSPM_Integrated_Analysis.csv`: summary analysis reference
- `models/rspm_models.pkl`: trained model bundle
- `scripts/`: engineering, dataset, model, and reporting utilities

## Local Run

Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

Start the app:

```powershell
streamlit run .\streamlit_app.py
```

Or use the included launcher:

```powershell
.\run_dashboard.bat
```

## Deploy Options

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the app file to `streamlit_app.py`.
4. Deploy.

### Render

This repo now includes [render.yaml](/Users/hp/OneDrive/Desktop/RSPM_Project/render.yaml), so Render can deploy it as a Blueprint with the correct build and start commands.

1. Push this repository to GitHub.
2. In Render, choose `New +` then `Blueprint`.
3. Connect the repository.
4. Render will detect `render.yaml` and create the web service automatically.
5. Deploy the service.

If you prefer manual setup instead of Blueprint, use:

- Environment: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

### Docker

Build the image:

```powershell
docker build -t rspm-dashboard .
```

Run the container:

```powershell
docker run -p 8501:8501 rspm-dashboard
```

## Upload Format

Uploaded CSVs should contain one row per scenario with these feature columns:

- `Reservoir_Pressure_psi`
- `Reservoir_Temperature_C`
- `Oil_Production_Rate_bbl_day`
- `Gas_Production_Rate_MSCF_day`
- `Water_Cut_percent`
- `Pipeline_Diameter_m`
- `Wall_Thickness_mm`
- `Pipeline_Length_km`
- `Flow_Velocity_m_s`
- `Fluid_Density_kg_m3`
- `Fluid_Viscosity_cP`
- `Corrosion_Rate_mm_year`
- `Internal_Pressure_psi`
- `Temperature_Gradient_C_km`
- `Elevation_Change_m`
- `Pipeline_Age_years`

`Calculated_Failure_Pressure_psi` is optional because the app can derive it automatically.

## Notes

- `RSPM_Integrated_Analysis.csv` is a summary file, not a prediction input dataset.
- Generated reports are written to `reports/`.
- On Render, the filesystem is ephemeral, so generated reports are temporary and recreated at runtime.
