import pandas as pd
from backend.core.config import ID_LIKE_UNIQUE_RATIO, CATEGORICAL_UNIQUE_THRESHOLD  

def load_csv(path) -> pd.DataFrame:
    """Loads CSV and ensures it contains data."""
    df = pd.read_csv(path)
    if df.empty or df.shape[1] == 0:
        raise ValueError("Uploaded file is not a valid CSV or is empty.")
    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    """Splits columns into numeric and categorical lists."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def drop_id_like_columns(df: pd.DataFrame, protected_columns: set[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Drops columns where every single row has a unique identifier."""
    protected_columns = protected_columns or set()
    dropped = []
    for col in df.columns:
        if col not in protected_columns and len(df) > 0 and (df[col].nunique(dropna=False) / len(df) >= ID_LIKE_UNIQUE_RATIO):
            dropped.append(col)
    clean_df = df.drop(columns=dropped)
    return clean_df, dropped

def detect_target_and_task(df: pd.DataFrame, target_column: str | None, task_type: str | None) -> tuple[str, str]:
    """Infers the target column and task type (classification/regression) if not supplied."""
    if target_column and target_column in df.columns:
        if task_type:
            return target_column, task_type
        if df[target_column].dtype == "object" or df[target_column].nunique() <= CATEGORICAL_UNIQUE_THRESHOLD:
            return target_column, "classification"
        return target_column, "regression"

    types = detect_column_types(df)
    for col in types["categorical"]:
        if df[col].nunique() <= CATEGORICAL_UNIQUE_THRESHOLD:
            return col, "classification"

    if types["numeric"]:
        return types["numeric"][-1], "regression"

    raise ValueError("Could not auto-detect a target column. Please specify one.")