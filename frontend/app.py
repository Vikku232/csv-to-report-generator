import gradio as gr
import requests
import tempfile
from pathlib import Path

BACKEND_URL = "http://localhost:8000/api"

def get_columns(file):
    import pandas as pd
    if file is None:
        return gr.update(choices=[], value=None)
    df = pd.read_csv(file.name, nrows=5)
    return gr.update(choices=df.columns.tolist(), value=None)

def generate_report(file, target_column, task_type):
    if file is None:
        raise gr.Error("Please upload a CSV file first.")

    with open(file.name, "rb") as f:
        files = {"file": (Path(file.name).name, f, "text/csv")}
        data = {}
        if target_column:
            data["target_column"] = target_column
        if task_type and task_type != "auto":
            data["task_type"] = task_type

        try:
            response = requests.post(
                f"{BACKEND_URL}/generate-report",
                files=files,
                data=data,
                timeout=300,
            )
        except requests.RequestException as exc:
            raise gr.Error(f"Backend is not reachable: {exc}") from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise gr.Error(f"Could not generate report: {detail}")

    out_path = Path(tempfile.gettempdir()) / "report.pdf"
    out_path.write_bytes(response.content)
    return str(out_path)

def check_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json()
    except Exception as e:
        return {"status": "backend unreachable", "error": str(e)}

with gr.Blocks(title="CSV Report Generator") as demo:
    gr.Markdown("## CSV → PDF Report Generator")

    with gr.Row():
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
        target_dropdown = gr.Dropdown(label="Target column (optional)", choices=[])
        task_radio = gr.Radio(["auto", "classification", "regression"], value="auto", label="Task type")

    csv_input.change(fn=get_columns, inputs=csv_input, outputs=target_dropdown)

    generate_btn = gr.Button("Generate Report", variant="primary")
    pdf_output = gr.File(label="Download Report")

    generate_btn.click(fn=generate_report, inputs=[csv_input, target_dropdown, task_radio], outputs=pdf_output)

    gr.Markdown("---")
    health_btn = gr.Button("Check Backend Health")
    health_output = gr.JSON()
    health_btn.click(fn=check_health, outputs=health_output)

if __name__ == "__main__":
    demo.launch(server_port=7860)