import os
import re
from pathlib import Path

import requests


def _load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name.strip(), value)


_load_project_env()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def _available_model(api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(GROQ_MODELS_URL, headers=headers, timeout=30)
    if not response.ok:
        raise ValueError(f"Groq model access failed with HTTP {response.status_code}: {response.text[:300]}")
    models = [item.get("id") for item in response.json().get("data", []) if item.get("id")]
    if not models:
        raise ValueError("Groq returned no available models for this API key.")
    if GROQ_MODEL in models:
        return GROQ_MODEL
    blocked = ("prompt-guard", "safeguard", "whisper", "orpheus")
    preferred = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/")
    candidates = [model for model in models if not any(name in model.lower() for name in blocked)]
    return next((model for model in candidates if model in preferred), candidates[0] if candidates else models[0])


def _factual_fallback(data_bundle: dict) -> str:
    features = ", ".join(feature for feature, _ in data_bundle.get("top_features", [])[:5]) or "the available features"
    if data_bundle["task_type"] == "classification":
        performance = f"test accuracy is {data_bundle.get('test_accuracy', 0):.4f}"
    else:
        performance = f"R2 is {data_bundle.get('r2', 0):.4f} and RMSE is {data_bundle.get('rmse', 0):.4f}"
    return (
        f"1. Executive Summary & Data Integrity: The dataset contains {data_bundle['rows']:,} rows and "
        f"{data_bundle['cols']} columns. Missing values account for {data_bundle['overall_missing_pct']:.2f}% "
        f"of the data, and the selected target is '{data_bundle['target_column']}'.\n\n"
        f"2. Baseline Predictive Performance & Reliability: The Random Forest baseline {performance}. "
        f"These results should be confirmed with cross-validation before deployment.\n\n"
        f"3. Key Feature Drivers & Domain Interpretation: The highest-ranked measured features are {features}. "
        f"They provide a focused starting point for feature review and further analysis.\n\n"
        f"4. Dimensionality & Cluster Analysis: PCA and clustering summarize the available feature structure "
        f"and provide a compact view of possible group separation.\n\n"
        f"5. Strategic Roadmap & Production Recommendations: Validate performance with cross-validation, "
        f"monitor data drift, improve feature quality, and compare additional models before production use."
    )


def generate_llm_summary(data_bundle: dict) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _factual_fallback(data_bundle)

    feature_names = ", ".join(feature for feature, _ in data_bundle.get("top_features", [])[:5]) or "none available"
    metrics = (
        f"test accuracy={data_bundle.get('test_accuracy')}, ROC AUC={data_bundle.get('roc_auc')}"
        if data_bundle["task_type"] == "classification"
        else f"R2={data_bundle.get('r2')}, RMSE={data_bundle.get('rmse')}"
    )
    prompt = f"""You are a Principal Data Scientist and Enterprise AI Strategist. Create a detailed executive report using only the measured facts below.
Return exactly five numbered sections with these headings, using HTML <b>, <br/>, and &bull; tags:
1. Executive Summary & Data Integrity
2. Baseline Predictive Performance & Reliability
3. Key Feature Drivers & Domain Interpretation
4. Dimensionality & Cluster Analysis
5. Strategic Roadmap & Production Recommendations
Write 180-240 words total so the complete summary fits cleanly in the report. Do not invent values, datasets, models, or claims.

Dataset: {data_bundle['dataset_name']}
Rows: {data_bundle['rows']}
Columns: {data_bundle['cols']}
Task: {data_bundle['task_type']}
Target: {data_bundle['target_column']}
Missing percentage: {data_bundle['overall_missing_pct']:.2f}%
Top features: {feature_names}
Model metrics: {metrics}
"""
    try:
        selected_model = _available_model(api_key)
    except (requests.RequestException, ValueError):
        return _factual_fallback(data_bundle)

    def request_summary(model: str) -> requests.Response:
        return requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "user", "content": "You are a precise data analyst. Use only supplied facts.\n\n" + prompt},
                ],
            },
            timeout=120,
        )

    try:
        response = request_summary(selected_model)
        if not response.ok:
            detail = response.text[:500]
            raise ValueError(f"Groq API returned HTTP {response.status_code}: {detail}")
        content = response.json()["choices"][0]["message"]["content"].strip()
        word_count = len(re.findall(r"[A-Za-z]{3,}", content))
        if len(content) < 200 or word_count < 30 or not re.search(r"\b[1-5][.)]", content):
            raise ValueError("Groq returned an invalid executive summary instead of the requested analysis.")
    except ValueError:
        return _factual_fallback(data_bundle)
    except requests.RequestException as exc:
        return _factual_fallback(data_bundle)
    except (KeyError, IndexError, TypeError) as exc:
        return _factual_fallback(data_bundle)

    if not content:
        return _factual_fallback(data_bundle)
    return content