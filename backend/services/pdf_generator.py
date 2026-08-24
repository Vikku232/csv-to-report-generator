# backend/services/pdf_generator.py
from datetime import datetime
from pathlib import Path
import re
from matplotlib.font_manager import FontProperties, findfont
import pandas as pd
from fpdf import FPDF

def _summary_sections(data_bundle: dict) -> list[tuple[str, str]]:
    task_type = data_bundle["task_type"].title()
    sections = [
        (
            "Dataset & Quality Assessment",
            f"The dataset contains {data_bundle['rows']} samples across {data_bundle['cols']} attributes. "
            f"Data quality includes {data_bundle['overall_missing_pct']:.2f}% missing values, making it suitable "
            f"for predictive modeling targeting '{data_bundle['target_column']}'.",
        ),
    ]
    if data_bundle.get("top_features"):
        features = ", ".join(feature for feature, _ in data_bundle["top_features"][:3])
    else:
        features = "the available features"

    if data_bundle["task_type"] == "classification":
        test_accuracy = data_bundle.get("test_accuracy")
        roc_auc = data_bundle.get("roc_auc")
        accuracy_text = f"{test_accuracy:.4f}" if test_accuracy is not None else "N/A"
        auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "N/A for multiclass data"
        baseline_text = (
            f"The Random Forest baseline achieved {accuracy_text} test accuracy and ROC AUC of {auc_text}. "
            f"The most influential predictive indicators are {features}."
        )
    else:
        baseline_text = (
            f"The Random Forest baseline achieved an R2 score of {data_bundle['r2']:.4f} "
            f"with RMSE of {data_bundle['rmse']:.4f}. The most influential predictive indicators are {features}."
        )

    sections.extend([
        (
            "Machine Learning Baseline",
            baseline_text,
        ),
        (
            "Dimensionality & Clustering",
            "PCA projection and clustering were included to summarize feature structure and identify natural groups in the data.",
        ),
        (
            "Strategic Next Steps",
            f"Focus future iterations on feature engineering around {features}, validate performance with cross-validation, "
            f"and evaluate additional {task_type.lower()} algorithms for further improvement.",
        ),
    ])
    return sections

class CleanReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("DejaVu", "", findfont(FontProperties(family="DejaVu Sans")))
        self.add_font("DejaVu", "B", findfont(FontProperties(family="DejaVu Sans", weight="bold")))
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("DejaVu", "B", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, "Automated Data & Model Profiling Report", align="R", ln=True)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")

    def page_heading(self, title: str):
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(33, 37, 41)
        self.cell(0, 8, title, ln=True)
        self.ln(1)

    def report_title(self, title: str, subtitle: str, generated_at: str):
        self.set_font("DejaVu", "B", 14)
        self.set_text_color(31, 78, 121)
        self.multi_cell(0, 8, title, align="C")
        self.ln(3)
        self.set_font("DejaVu", "B", 13)
        self.set_text_color(33, 37, 41)
        self.cell(0, 7, subtitle, align="C", ln=True)
        self.ln(4)
        badge_w = 108
        badge_h = 9
        badge_x = (self.w - badge_w) / 2
        badge_y = self.get_y()
        self.set_fill_color(232, 242, 252)
        self.set_draw_color(31, 78, 121)
        self.set_line_width(0.45)
        self.rect(badge_x, badge_y, badge_w, badge_h, style="DF")
        self.set_xy(badge_x, badge_y + 2)
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(31, 78, 121)
        self.cell(badge_w, 5, f"Generated on: {generated_at}", align="C")
        self.set_y(badge_y + badge_h + 9)

    def section_subheading(self, title: str):
        self.set_font("DejaVu", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, title, ln=True)
        self.ln(1)

    def key_value_block(self, items: list[tuple[str, str]]):
        self.set_font("DejaVu", "", 9)
        for label, val in items:
            self.set_font("DejaVu", "B", 9)
            self.set_text_color(60, 60, 60)
            self.cell(50, 5, f"{label}:", ln=False)
            self.set_font("DejaVu", "", 9)
            self.set_text_color(20, 20, 20)
            self.cell(0, 5, str(val), ln=True)
        self.ln(2)

    def add_table(self, df: pd.DataFrame, col_widths: list[float] | None = None, max_rows: int = 20):
        self.set_font("DejaVu", "B", 8)
        self.set_fill_color(31, 78, 121)
        self.set_text_color(30, 30, 30)

        n_cols = len(df.columns)
        if not col_widths:
            w = 180.0 / max(n_cols, 1)
            col_widths = [w] * n_cols

        for i, col in enumerate(df.columns):
            self.set_text_color(255, 255, 255)
            self.cell(col_widths[i], 6, str(col)[:20], border=1, fill=True, align="C")
        self.ln()

        self.set_font("DejaVu", "", 8)
        self.set_text_color(50, 50, 50)
        for row_index, (_, row) in enumerate(df.head(max_rows).iterrows()):
            self.set_fill_color(235, 245, 250) if row_index % 2 == 0 else self.set_fill_color(250, 250, 250)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5.5, str(val)[:20], border=1, fill=True, align="C")
            self.ln()
        self.ln(3)

    def add_chart(self, img_path: Path, w: int = 175):
        if img_path and Path(img_path).exists():
            self.image(str(img_path), x=self.get_x(), w=w)
            self.ln(3)

    def add_summary(self, sections: list[tuple[str, str]]):
        card_x = self.l_margin + 1
        card_w = self.w - self.l_margin - self.r_margin - 2
        for index, (heading, body) in enumerate(sections, start=1):
            inner_w = card_w - 16
            self.set_font("DejaVu", "", 9)
            body_lines = self._wrap_pdf_text(body, inner_w)
            self.set_font("Helvetica", "B", 9)
            heading_lines = self._wrap_pdf_text(f"{index}. {heading}:", inner_w)
            card_h = 10 + (len(heading_lines) + max(1, len(body_lines))) * 4.8 + 8
            if self.get_y() + card_h > self.h - self.b_margin:
                self.add_page()
                self.set_font("DejaVu", "B", 14)
                self.set_text_color(31, 78, 121)
                self.cell(0, 8, "AI-Generated Executive Summary & Strategic Insights", align="C", ln=True)
                self.ln(5)
            card_y = self.get_y()
            self.set_fill_color(248, 250, 252)
            self.set_draw_color(215, 220, 228)
            self.set_line_width(0.3)
            self.rect(card_x, card_y, card_w, card_h, style="DF")
            self.set_fill_color(31, 78, 121)
            self.rect(card_x, card_y, 3, card_h, style="F")
            self.set_xy(card_x + 8, card_y + 5)
            self.set_text_color(25, 35, 60)
            self.set_font("DejaVu", "B", 9)
            self.multi_cell(inner_w, 4.8, f"{index}. {heading}:", align="L")
            self.set_font("DejaVu", "", 9)
            self.set_x(card_x + 8)
            self.multi_cell(inner_w, 4.8, body, align="L")
            self.set_y(card_y + card_h + 5)

    def _wrap_pdf_text(self, text: str, width: float) -> list[str]:
        lines = []
        for paragraph in text.splitlines() or [text]:
            words = paragraph.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and self.get_string_width(candidate) > width:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
        return lines

    def add_summary_text(self, text: str):
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\*{1,2}", "", text)
        text = text.replace("&bull;", "- ").replace("&amp;", "&")
        text = text.encode("ascii", "replace").decode("ascii")
        matches = re.findall(r"(?ms)^\s*(\d+)\.\s*([^:]+):\s*(.*?)(?=^\s*\d+\.\s|\Z)", text)
        sections = [(heading.strip(), body.strip()) for _, heading, body in matches]
        if not sections:
            sections = [("Executive Insight", text.strip())]
        self.add_summary(sections)


def generate_report_pdf(data_bundle: dict, output_path: Path) -> Path:
    pdf = CleanReportPDF()

    # Overview Page
    pdf.add_page()
    generated_at = datetime.now().strftime("%A, %B %d, %Y • %I:%M:%S %p")
    pdf.report_title(
        "Automated Exploratory Data Analysis, Machine Learning Baseline & Strategic AI Insights",
        f"{data_bundle.get('dataset_name', 'Dataset')} Report",
        generated_at,
    )
    
    meta_info = [
        ("Generated on", generated_at),
        ("Task type", data_bundle["task_type"]),
        ("Target column", data_bundle["target_column"]),
        ("Rows", str(data_bundle["rows"])),
        ("Columns", str(data_bundle["cols"])),
        ("Numeric columns", str(len(data_bundle["numeric_cols"]))),
        ("Categorical columns", str(len(data_bundle["categorical_cols"]))),
        ("Overall missing", f"{data_bundle['overall_missing_pct']:.2f}%"),
        ("ID-like columns removed", f"{len(data_bundle['dropped_ids'])} ({', '.join(data_bundle['dropped_ids']) or 'None'})")
    ]
    pdf.key_value_block(meta_info)

    pdf.ln(6)
    pdf.section_subheading("Top 5 Columns by Missing Values")
    pdf.add_table(data_bundle["top_missing_df"], col_widths=[40, 30, 30, 40, 40])

    pdf.add_page()
    pdf.page_heading("Missing Values Report")
    pdf.add_table(data_bundle["missing_df"], col_widths=[65, 55, 60], max_rows=1000)

    if data_bundle.get("top_features"):
        pdf.section_subheading("Top Influencing Features")
        pdf.key_value_block([(feat, f"{score:.4f}") for feat, score in data_bundle["top_features"]])

    # Numeric Heatmap Pages
    for heatmap_img in data_bundle.get("numeric_heatmap_imgs", []):
        pdf.add_page()
        pdf.page_heading("Numeric Summary")
        pdf.add_chart(heatmap_img, w=180)

    # Correlation Matrix
    if data_bundle.get("corr_img"):
        pdf.add_page()
        pdf.page_heading("Correlation Heatmap")
        pdf.add_chart(data_bundle["corr_img"], w=180)

    # Categorical Summary & Target Distribution
    if data_bundle.get("cat_summary_df") is not None and not data_bundle["cat_summary_df"].empty:
        pdf.add_page()
        pdf.page_heading("Categorical Columns Summary")
        pdf.add_table(data_bundle["cat_summary_df"], col_widths=[45, 30, 30, 45, 30])
        if data_bundle.get("target_img"):
            pdf.section_subheading("Target Distribution")
            pdf.add_chart(data_bundle["target_img"], w=150)

    # KDE Grids
    for kde_img in data_bundle.get("kde_imgs", []):
        pdf.add_page()
        pdf.add_chart(kde_img, w=180)

    # Outlier Proportion
    if data_bundle.get("outlier_img"):
        pdf.add_page()
        pdf.page_heading("Outlier Proportion")
        pdf.add_chart(data_bundle["outlier_img"], w=180)

    # Duplicate Distribution
    if data_bundle.get("dup_img"):
        pdf.add_page()
        pdf.page_heading("Duplicate Distribution")
        pdf.add_chart(data_bundle["dup_img"], w=180)

    # Feature Importance
    if data_bundle.get("fi_img"):
        pdf.add_page()
        pdf.page_heading("Feature Importance")
        pdf.add_chart(data_bundle["fi_img"], w=180)

    # PCA Clustering
    if data_bundle.get("pca_img"):
        pdf.add_page()
        pdf.page_heading("PCA Clusters")
        pdf.add_chart(data_bundle["pca_img"], w=170)

    # Predicted vs Actual
    if data_bundle.get("pred_img"):
        pdf.add_page()
        pdf.page_heading("Predicted Probability vs Actual")
        pdf.add_chart(data_bundle["pred_img"], w=170)

    # Model Summary
    if data_bundle.get("model_summary_img"):
        pdf.add_page()
        pdf.page_heading("Model Summary & ROC/Residuals")
        pdf.add_chart(data_bundle["model_summary_img"], w=180)

    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(31, 78, 121)
    pdf.cell(0, 10, "AI-Generated Executive Summary & Strategic Insights", align="C", ln=True)
    pdf.ln(7)
    pdf.add_summary_text(data_bundle["llm_summary"])

    pdf.output(str(output_path))
    return output_path