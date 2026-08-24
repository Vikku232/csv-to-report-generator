# backend/services/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, r2_score, root_mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from backend.core.config import RANDOM_STATE

MAX_ONE_HOT_CATEGORIES = 50
MAX_PCA_ROWS = 5000

def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    source = df[feature_cols].copy()
    numeric_features = source.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in source.columns if column not in numeric_features]

    numeric = source[numeric_features].copy()
    for column in numeric.columns:
        median = numeric[column].median()
        numeric[column] = numeric[column].fillna(0 if pd.isna(median) else median)

    low_cardinality = []
    encoded_high_cardinality = pd.DataFrame(index=source.index)
    for column in categorical_features:
        values = source[column]
        unique_count = values.nunique(dropna=False)
        if unique_count <= MAX_ONE_HOT_CATEGORIES:
            low_cardinality.append(column)
        else:
            frequencies = values.value_counts(dropna=False, normalize=True)
            encoded_high_cardinality[f"{column}__frequency"] = values.map(frequencies).fillna(0)

    one_hot = pd.get_dummies(source[low_cardinality], drop_first=True, dtype=float) if low_cardinality else pd.DataFrame(index=source.index)
    features = pd.concat([numeric, encoded_high_cardinality, one_hot], axis=1)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

def train_baseline_model(df: pd.DataFrame, target_column: str, task_type: str, numeric_cols: list[str]) -> dict:
    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise ValueError("Dataset must contain at least one feature column besides the target column.")

    X = _prepare_features(df, feature_cols)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing. Keep at least one non-ID feature column.")

    y = df[target_column]
    valid_rows = y.notna()
    if not valid_rows.all():
        X = X.loc[valid_rows]
        y = y.loc[valid_rows]
    if len(y) < 5:
        raise ValueError("The target column must contain at least 5 non-empty rows.")

    label_encoder = None
    if task_type == "classification":
        if y.dtype == "object" or isinstance(y.iloc[0], str):
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    stratify = y if (task_type == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)

    model.fit(X_train, y_train)

    result = {
        "model": model,
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "label_encoder": label_encoder,
    }

    if task_type == "classification":
        y_pred = model.predict(X_test)
        result["train_accuracy"] = accuracy_score(y_train, model.predict(X_train))
        result["test_accuracy"] = accuracy_score(y_test, y_pred)
        if len(np.unique(y)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            result["roc_auc"] = roc_auc_score(y_test, y_proba)
            result["y_proba"] = y_proba
            result["fpr"], result["tpr"], _ = roc_curve(y_test, y_proba)
    else:
        y_pred = model.predict(X_test)
        result["r2"] = r2_score(y_test, y_pred)
        result["rmse"] = root_mean_squared_error(y_test, y_pred)
        result["y_pred"] = y_pred

    return result

def get_feature_importance(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

def get_pca_clusters(X: pd.DataFrame, k: int = 2) -> dict:
    if X.shape[0] < 2 or X.shape[1] < 2:
        return {"pc1": np.zeros(len(X)), "pc2": np.zeros(len(X)), "labels": np.zeros(len(X), dtype=int), "silhouette": 0.0}

    pca_input = X.sample(MAX_PCA_ROWS, random_state=RANDOM_STATE) if len(X) > MAX_PCA_ROWS else X
    X_scaled = StandardScaler().fit_transform(pca_input)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=min(k, len(coords)), random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(coords)
    score = silhouette_score(coords, labels) if len(set(labels)) > 1 else 0.0
    return {"pc1": coords[:, 0], "pc2": coords[:, 1], "labels": labels, "silhouette": round(float(score), 3)}