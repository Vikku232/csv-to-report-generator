# backend/api/routes/report.py
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from backend.utils.file_utils import save_uploaded_file, new_output_path, cleanup_file
from backend.services import data_loader, eda_analyzer, model_trainer, visualizer
from backend.services.llm_summary import generate_llm_summary
from backend.services.pdf_generator import generate_report_pdf

router = APIRouter()

@router.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    task_type: str | None = Form(None),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    upload_path = save_uploaded_file(file)
    try:
        df = data_loader.load_csv(upload_path)
    except Exception as e:
        cleanup_file(upload_path)
        raise HTTPException(status_code=400, detail=str(e))

    if target_column and target_column not in df.columns:
        cleanup_file(upload_path)
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' was not found in the CSV.")

    raw_rows, raw_cols = df.shape
    if raw_rows < 5:
        cleanup_file(upload_path)
        raise HTTPException(status_code=400, detail="Please provide at least 5 data rows for reliable report generation.")

    target_column, task_type = data_loader.detect_target_and_task(df, target_column, task_type)
    df, dropped_ids = data_loader.drop_id_like_columns(
        df,
        protected_columns={target_column},
    )
    types = data_loader.detect_column_types(df)
    numeric_cols, categorical_cols = types["numeric"], types["categorical"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        missing_df = eda_analyzer.missing_report(df)
        corr_df = eda_analyzer.correlation_matrix(df, numeric_cols)
        outlier_df = eda_analyzer.outlier_report(df, numeric_cols)
        dup_df = eda_analyzer.duplicate_report(df)
        target_stats = eda_analyzer.target_distribution_stats(df, target_column)
        cat_summary_df = eda_analyzer.categorical_summary(df, categorical_cols)

        try:
            model_result = model_trainer.train_baseline_model(df, target_column, task_type, numeric_cols)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        fi_df = model_trainer.get_feature_importance(model_result["model"], model_result["feature_names"])
        pca_result = model_trainer.get_pca_clusters(model_result["X"])

        top_missing = missing_df.head(5).copy()
        top_missing["dtype"] = [str(df[c].dtype) for c in top_missing["column"]]
        top_missing["unique"] = [df[c].nunique() for c in top_missing["column"]]
        top_missing = top_missing[["column", "dtype", "unique", "missing_count", "missing_percent"]]

        top_features = list(zip(fi_df["feature"], fi_df["importance"]))[:5]

        data_bundle = {
            "dataset_name": Path(file.filename).stem,
            "task_type": task_type,
            "target_column": target_column,
            "rows": raw_rows,
            "cols": raw_cols,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "overall_missing_pct": float(df.isnull().mean().mean() * 100),
            "dropped_ids": dropped_ids,
            "missing_df": missing_df,
            "top_missing_df": top_missing,
            "top_features": top_features,
            "train_accuracy": model_result.get("train_accuracy"),
            "test_accuracy": model_result.get("test_accuracy"),
            "roc_auc": model_result.get("roc_auc"),
            "r2": model_result.get("r2"),
            "rmse": model_result.get("rmse"),
            "numeric_heatmap_imgs": visualizer.plot_numeric_summary_heatmaps(df, numeric_cols, tmp_dir),
            "corr_img": visualizer.plot_correlation_heatmap(corr_df, tmp_dir) if not corr_df.empty else None,
            "cat_summary_df": cat_summary_df.reset_index().rename(columns={"index": "column"}) if not cat_summary_df.empty else None,
            "target_img": visualizer.plot_target_distribution(target_stats["value_counts"], target_column, tmp_dir),
            "kde_imgs": visualizer.plot_kde_grid(df, numeric_cols, tmp_dir),
            "outlier_img": visualizer.plot_outlier_proportion(outlier_df, tmp_dir),
            "dup_img": visualizer.plot_duplicate_distribution(dup_df, tmp_dir),
            "fi_img": visualizer.plot_feature_importance(fi_df, target_column, tmp_dir),
            "pca_img": visualizer.plot_pca_clusters(pca_result, tmp_dir),
            "pred_img": visualizer.plot_predicted_vs_actual(
                model_result["y_test"],
                model_result.get("y_proba", model_result.get("y_pred")),
                task_type,
                tmp_dir
            ),
            "model_summary_img": visualizer.plot_model_summary_and_roc(model_result, task_type, tmp_dir),
            "roc_img": visualizer.plot_roc_curve(model_result, tmp_dir),
        }

        try:
            data_bundle["llm_summary"] = generate_llm_summary(data_bundle)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        output_path = new_output_path("report.pdf")
        generate_report_pdf(data_bundle, output_path)

    cleanup_file(upload_path)
    return FileResponse(output_path, media_type="application/pdf", filename="report.pdf")