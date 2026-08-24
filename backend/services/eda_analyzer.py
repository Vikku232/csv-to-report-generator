# backend/services/eda_analyzer.py
import pandas as pd
import numpy as np

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    out = pd.DataFrame({"column": df.columns, "missing_count": miss.values, "missing_percent": pct.values})
    return out.sort_values("missing_count", ascending=False).reset_index(drop=True)

def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].describe().T.round(2)

def categorical_summary(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    if not categorical_cols:
        return pd.DataFrame()
    return df[categorical_cols].describe().T

def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr()

def outlier_report(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Computes outlier percentage using 3-sigma and IQR methods."""
    columns = ["column", "pct_outliers_3sigma", "pct_outliers_iqr"]
    if not numeric_cols:
        return pd.DataFrame(columns=columns)

    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        mean, std = series.mean(), series.std()
        sigma_outliers = ((series - mean).abs() > 3 * std).sum() if std > 0 else 0

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_outliers = ((series < lower) | (series > upper)).sum()

        rows.append({
            "column": col,
            "pct_outliers_3sigma": round(sigma_outliers / len(series) * 100, 2),
            "pct_outliers_iqr": round(iqr_outliers / len(series) * 100, 2),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("pct_outliers_iqr", ascending=False).reset_index(drop=True)

def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Counts duplicate frequency per column."""
    rows = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        dup_count = counts[counts > 1].sum()
        rows.append({"column": col, "duplicate_value_count": int(dup_count)})
    return pd.DataFrame(rows).sort_values("duplicate_value_count", ascending=False).reset_index(drop=True)

def target_distribution_stats(df: pd.DataFrame, target_column: str) -> dict:
    vc = df[target_column].value_counts()
    return {
        "value_counts": vc,
        "count": len(df[target_column].dropna()),
        "unique": df[target_column].nunique(),
        "top": vc.index[0] if len(vc) > 0 else None,
        "freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
    }