# 📊 AutoCSV • Automated CSV Report Generator & AutoML Studio

An end-to-end Python system for automated **Exploratory Data Analysis (EDA)**, **AutoML multi-model benchmarking**, and **executive multi-page PDF report generation** powered by **FastAPI**, **ReportLab**, and **Gradio**.

---

## 🚀 Key Features

- **Multi-Format Ingestion**: Supports `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls` with automatic encoding detection (`utf-8`, `latin1`, `cp1252`), delimiter sniffing, and built-in benchmark datasets (*Titanic Survival*, *California Housing*, *Iris Morphology*, *Customer Churn*, *Retail Store Sales*).
- **Comprehensive EDA Engine**:
  - Inferred semantic types (numeric, categorical, datetime, boolean, text, ID).
  - Statistical distributions (Mean, Std, Median, IQR, Skewness, Kurtosis, Outlier counts).
  - Outlier detection via Interquartile Range (IQR).
  - Pearson correlation matrix.
  - Data Quality Health Score calculation (0–100%).
  - Automated natural language insights & actionable data hygiene recommendations.
- **AutoML Benchmark Zoo**:
  - Automatic task type detection (Classification vs. Regression).
  - Robust preprocessing pipeline (Median/Mode imputation, One-Hot encoding, StandardScaler).
  - Multi-model evaluation with Stratified/K-Fold Cross-Validation:
    - *Classification*: Random Forest, Gradient Boosting, Logistic Regression, Decision Tree, Extra Trees, KNN.
    - *Regression*: Random Forest, Gradient Boosting, Ridge, Linear Regression, Decision Tree, Extra Trees.
  - Metrics computation (Accuracy, F1-Score, Precision, Recall, ROC-AUC, R², RMSE, MAE, MAPE).
  - Leaderboard ranking, Feature Importance ranking, Confusion Matrix heatmaps, and Residual analysis.
- **Executive PDF Report Generator**:
  - Publication-grade multi-page PDF generation via ReportLab Platypus.
  - Two-pass `NumberedCanvas` with running headers and dynamic `"Page X of Y"` footers.
  - KPI summary badges, data dictionary tables, embedded high-resolution Seaborn charts, AutoML leaderboards, and strategic recommendations.
- **Modern Interactive UI**:
  - Tabbed Gradio dashboard with KPI cards, interactive data preview, distribution histograms, outlier boxplots, heatmap explorer, and 1-click PDF export.

---

## 📂 Project Architecture

```
csv-report-generator/
│
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entrypoint
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            # Settings, paths, color palette
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic API schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # File loader & sample dataset factory
│   │   ├── eda_analyzer.py      # Statistical profiler & insights engine
│   │   ├── model_trainer.py     # AutoML training & cross-validation
│   │   ├── visualizer.py        # High-DPI Seaborn/Matplotlib chart generator
│   │   └── pdf_generator.py     # Multi-page ReportLab PDF compiler
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py        # System health & storage stats
│   │       └── report.py        # Upload, preview, EDA, train, PDF endpoints
│   │
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py        # File validation, hashing, storage tools
│
├── frontend/
│   ├── __init__.py
│   └── app.py                   # Main Gradio application interface & entrypoint
│
├── storage/
│   ├── uploads/                 # Temporary uploaded datasets
│   └── outputs/                 # Compiled PDF reports
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore configuration
└── README.md                    # Project documentation
```

---

## 🛠️ Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Combined Dashboard and API
```bash
set GROQ_API_KEY=your_rotated_groq_key
uvicorn backend.main:app --reload --port 8000
```
PowerShell:
```powershell
$env:GROQ_API_KEY = "your_rotated_groq_key"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*Access the dashboard at `http://127.0.0.1:8000`*

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_rotated_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
```
The API key is loaded from `.env` and is never committed because `.env` is gitignored.

*Access Interactive Swagger Documentation at `http://127.0.0.1:8000/docs`*

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and storage usage |
| `GET` | `/api/sample-datasets` | List available pre-packaged demo datasets |
| `POST` | `/api/load-sample/{sample_id}` | Load a demo dataset into workspace |
| `POST` | `/api/upload` | Upload `.csv`, `.tsv`, or `.xlsx` dataset file |
| `GET` | `/api/preview/{file_id}` | Preview top rows and summary schema |
| `POST` | `/api/eda/{file_id}` | Run full Exploratory Data Analysis |
| `POST` | `/api/train` | Run AutoML model benchmark with cross-validation |
| `POST` | `/api/generate-report` | Compile executive multi-page PDF intelligence report |
| `GET` | `/api/download/{report_id}` | Download generated PDF report file |
