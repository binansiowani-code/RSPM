# RSPM — South Sudan Field Guardian (PFG AI)

A Streamlit dashboard for pipeline integrity and leak prediction. The app trains a Random Forest model on pipeline segment data and provides interactive risk dashboards, AI recommendations, and CSV exports.

## Quick overview
- Project: `RSPM`
- App entrypoint: `app.py`
- Default dataset: `dataset_rspm_paloch.csv`
- Requirements: `requirements.txt`

## Features
- Train a Random Forest leak prediction model
- Interactive Plotly visualizations and risk map
- Segment inspector with AI-generated recommendations
- Export critical segments or full dataset as CSV

## Requirements
- Python 3.10+ recommended
- All Python packages listed in `requirements.txt`

## Quick Start (Windows PowerShell)

1. Clone the repository (if you haven't already):

```powershell
git clone https://github.com/binansiowani-code/RSPM.git
Set-Location RSPM
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run the app locally:

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

## Deployment (Streamlit Cloud)

1. Push your repository to GitHub (this repo is already linked):

```powershell
git add .
git commit -m "chore: update" || Write-Output "Nothing to commit"
git push
```

2. In Streamlit Cloud, connect the GitHub repo and choose the `main` branch. Streamlit Cloud will install packages from `requirements.txt` automatically.

3. If the app is already deployed, trigger a redeploy from the app's dashboard (Manage app → Deploy) after pushing changes.

## Configuration
- Default dataset path is set in `app.py` via `Config.DEFAULT_DATA_PATH` (value: `dataset_rspm_paloch.csv`).
- You can upload a CSV from the sidebar or provide a file path when running locally or deploying.

## File Structure
- `app.py` — Streamlit app
- `dataset_rspm_paloch.csv` — sample dataset (committed here; consider adding to `.gitignore` for large/production data)
- `requirements.txt` — Python dependencies

## Notes & Next Steps
- Consider adding a `.gitignore` to avoid committing large datasets or virtual environments.
- Pin versions in `requirements.txt` for reproducible deployments.

## Contributing
Feel free to open issues or pull requests. For quick changes, commit on a branch and open a PR against `main`.

## License
Specify a license if you intend to open-source this project (e.g., MIT). I can add a `LICENSE` file if you want.
"# RSPM" 
"# RSPM" 
"# RSPM" 
"# RSPM" 
"# RSPM" 
