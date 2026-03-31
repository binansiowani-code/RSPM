# RSPM Command Center

RSPM Command Center is a Streamlit-based dashboard for Reservoir and Surface Pipeline Monitoring. It combines engineering calculations, architecture-driven feature generation, machine learning risk diagnosis, and report export in one interface for evaluating pipeline integrity and leak severity.

## What the Project Does

- Monitors reservoir, production, and pipeline operating conditions in a single dashboard
- Calculates engineering indicators such as failure pressure, integrity margin, and failure probability
- Builds a 4-layer architecture flow from raw inputs to integrated features, leak risk scoring, and decision actions
- Runs a trained ML model for leak-risk diagnosis using the bundled model artifacts
- Supports scenario selection from the packaged dataset or custom CSV uploads
- Exports results as Excel and PDF reports

## 4-Layer Architecture

The application follows a structured pipeline:

1. Layer 1: raw reservoir and pipeline operating inputs
2. Layer 2: 22 integrated engineered features
3. Layer 3: AI outputs including classification and regression-based risk indicators
4. Layer 4: decision outputs mapped to `No Leak`, `Minor Leak`, `Moderate Leak`, and `Major Leak`

## Architecture Diagram

![Architecture diagram placeholder](./reports/architecture-diagram-placeholder.png)

_Replace this placeholder with the final architecture diagram image when it is available._

Recommended actions are:

- `No Leak` -> `Monitor`
- `Minor Leak` -> `Inspect`
- `Moderate Leak` -> `Intervene`
- `Major Leak` -> `Shut In Well`

## Project Structure

[app.py](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/app.py) is the deployment entrypoint and loads the dashboard module.

[dashboard/app.py](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/dashboard/app.py) contains the main Streamlit dashboard UI and workflow.

[scripts/architecture_engine.py](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/scripts/architecture_engine.py) builds integrated features, LRS scoring, and decision outputs.

[scripts/pipeline_model.py](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/scripts/pipeline_model.py) contains engineering calculations for failure pressure, failure probability, and health interpretation.

[scripts/report_generator.py](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/scripts/report_generator.py) creates Excel and PDF assessment reports.

[models/rspm_models.pkl](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/models/rspm_models.pkl) stores the trained model bundle used by the dashboard.

[data/rspm_integrated.csv](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/data/rspm_integrated.csv) is the main scenario dataset used in the dashboard.

[data/RSPM_Integrated_Analysis.csv](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/data/RSPM_Integrated_Analysis.csv) is a summary/reference file shown in the app.

[reports/](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/reports) stores generated report outputs.

## Requirements

- Python 3.13 recommended
- Streamlit and the packages listed in [requirements.txt](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/requirements.txt)

Install dependencies with:

```powershell
python -m pip install -r .\requirements.txt
```

## Running Locally

Start the dashboard directly:

```powershell
python -m streamlit run .\app.py
```

Or use one of the included launchers:

```powershell
.\run_dashboard.ps1
```

```bat
run_dashboard.bat
```

The app will open in Streamlit on the default local port, typically `8501`.

## Input Data Format

Uploaded CSV files should contain one row per scenario with these required columns:

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

`Calculated_Failure_Pressure_psi` is optional because the app can compute it automatically.

## Outputs

For each scenario, the dashboard can produce:

- failure pressure and failure probability estimates
- architecture LRS score
- leak severity decision class
- recommended operational action
- downloadable Excel and PDF reports

Generated reports are written to [reports/](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/reports).

## Deployment

### Render

The repository includes [render.yaml](/c:/Users/hp/OneDrive/Desktop/RSPM_Project/render.yaml) for deployment as a Render Blueprint.

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Docker

Build the image:

```powershell
docker build -t rspm-command-center .
```

Run the container:

```powershell
docker run -p 8501:8501 rspm-command-center
```

## Notes

- The dashboard uses the packaged model artifact in `models/rspm_models.pkl`
- `RSPM_Integrated_Analysis.csv` is a reference summary, not a prediction input file
- Render deployments use ephemeral storage, so generated report files are temporary
