import os
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

# Create directories
os.makedirs("12_FINAL_REPORT/assets", exist_ok=True)
os.makedirs("12_FINAL_REPORT/charts", exist_ok=True)
os.makedirs("12_FINAL_REPORT/diagrams", exist_ok=True)

# ---------------------------------------------------------
# 1. PDF GENERATION
# ---------------------------------------------------------
class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "EdgeVision PPE Compliance Platform - Final Submission Report", new_x="RIGHT", new_y="TOP", align="R")
            self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", new_x="RIGHT", new_y="TOP", align="C")

    def chapter_title(self, num, title):
        self.add_page()
        self.set_font("helvetica", "B", 24)
        self.set_text_color(15, 32, 67)
        self.cell(0, 20, f"SECTION {num}", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("helvetica", "B", 28)
        self.cell(0, 15, title.upper(), new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_line_width(1)
        self.set_draw_color(255, 140, 0)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def chapter_body(self, text, font_size=12, bold=False):
        style = "B" if bold else ""
        self.set_font("helvetica", style, font_size)
        self.set_text_color(50, 50, 50)
        self.multi_cell(190, 8, text)
        self.ln()
        
    def bullet_list(self, items):
        self.set_font("helvetica", "", 12)
        for item in items:
            self.multi_cell(190, 8, "- " + item)
        self.ln()

    def add_image(self, filepath, caption):
        if os.path.exists(filepath):
            self.image(filepath, x="C", w=150)
            self.set_font("helvetica", "I", 10)
            self.cell(0, 10, f"Figure: {caption}", new_x="LMARGIN", new_y="NEXT", align="C")
            self.ln(5)
            
    def add_table_row(self, col1, col2, col3, col4, bold=False):
        self.set_font("helvetica", "B" if bold else "", 10)
        self.cell(40, 10, col1, border=1)
        self.cell(40, 10, col2, border=1)
        self.cell(40, 10, col3, border=1)
        self.cell(60, 10, col4, border=1, new_x="LMARGIN", new_y="NEXT")

def generate_pdf():
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover Page
    pdf.add_page()
    pdf.set_y(80)
    pdf.set_font("helvetica", "B", 42)
    pdf.set_text_color(15, 32, 67)
    pdf.cell(0, 20, "EDGEVISION", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(255, 140, 0)
    pdf.cell(0, 15, "PPE COMPLIANCE & WORK-AT-HEIGHT SAFETY", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.set_font("helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.ln(20)
    pdf.cell(0, 10, "AI-POWERED EDGE COMPUTER VISION FOR INDUSTRIAL SAFETY", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 10, "Final Technical Submission Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 10, "Model: V3-HN Production Validated", new_x="LMARGIN", new_y="NEXT", align="C")
    
    pdf.add_image('12_FINAL_REPORT/diagrams/architecture.png', "System Architecture Overview")

    # Section 1
    pdf.chapter_title("1", "Executive Summary")
    pdf.chapter_body("The EdgeVision PPE Compliance Platform is an edge-deployed computer vision system designed to enforce industrial safety protocols in real-time. It monitors restricted zones, verifies required PPE (helmets and vests), and performs temporal validation to ensure high precision in violation alerting.\n\nThis final report details the successful promotion of the V3-HN YOLOv8 model, which utilizes a hard-negative mining strategy to eliminate false positives seen in early prototypes. The system integrates a robust FastAPI backend, a Next.js visualization dashboard, and is fully staged for TensorRT deployment on target Jetson hardware.")
    
    # Section 2
    pdf.chapter_title("2", "System Architecture")
    pdf.chapter_body("EdgeVision utilizes a modular microservices architecture designed for real-time edge processing. A visual overview of the data flow and system integration is shown below.")
    pdf.add_image('12_FINAL_REPORT/diagrams/architecture.png', "System Architecture Diagram")
    pdf.chapter_body("Deployment Pipeline Flow:")
    pdf.add_image('12_FINAL_REPORT/diagrams/deployment.png', "Deployment Pipeline")
    
    # Section 3
    pdf.chapter_title("3", "Dataset & Hard Negatives")
    pdf.chapter_body("The system is trained on the Construction-PPE dataset (17,452 train, 2,438 val, 2,464 test). During V2 testing, the model suffered from real-world false positives (e.g., detecting yellow pipes as helmets). To resolve this, the V3-HN dataset was enriched with visually similar background images (hard negatives) containing no labels. This forced the model to learn distinguishing features.")
    pdf.add_image('12_FINAL_REPORT/charts/dataset_split.png', "Dataset Split")
    pdf.add_image('12_FINAL_REPORT/charts/class_distribution.png', "Class Distribution")
    
    # Section 4
    pdf.chapter_title("4", "Model Development & Architecture")
    pdf.chapter_body("The final production model (V3-HN) architecture exactly matches the validated .pt metadata:")
    pdf.bullet_list([
        "Base Architecture: YOLOv8n (Nano)",
        "Layers: 130",
        "Parameters: 3,012,018",
        "Gradients: 0",
        "GFLOPs: 8.2",
        "Classes: 6 (person, helmet, no_helmet, vest, boots, harness)"
    ])
    
    pdf.chapter_body("\nModel Trade-off Decision:")
    pdf.chapter_body("V3-HN trades a modest reduction in helmet/vest recall for substantially improved real-world false-positive behavior and improved no-helmet recall. This is a critical requirement for production trust.", bold=True)
    
    # Section 5
    pdf.chapter_title("5", "Model Decision & Performance")
    pdf.chapter_body("The following table demonstrates the strict metric comparison used to select V3-HN for production and classify V3-BOOTS as experimental.")
    
    pdf.add_table_row("Metric", "V2", "V3-HN", "V3-BOOTS", bold=True)
    pdf.add_table_row("Role", "Rollback", "Production", "Experimental")
    pdf.add_table_row("mAP50", "84.45%", "84.20%", "80.10%")
    pdf.add_table_row("mAP50-95", "50.12%", "48.77%", "N/A")
    pdf.add_table_row("Helmet AP50", "90.15%", "88.65%", "82.20%")
    pdf.add_table_row("Vest AP50", "82.37%", "79.77%", "86.10%")
    pdf.add_table_row("Boots AP50", "N/A", "N/A", "80.10%")
    pdf.add_table_row("Boots Recall", "N/A", "N/A", "75.30%")
    pdf.add_table_row("Warm FPS", "24.3", "16.2", "27.82")
    pdf.add_table_row("Helmet FP", "154", "0", "N/A")
    pdf.add_table_row("Vest FP", "10", "0", "N/A")
    
    pdf.ln(10)
    pdf.chapter_body("V3-BOOTS is experimental only and is NOT the production model. V3-HN remains production. The Boots model suffers from bounding box inconsistencies.")
    
    pdf.add_image('12_FINAL_REPORT/charts/model_accuracy.png', "V2 vs V3-HN Accuracy")
    pdf.add_image('12_FINAL_REPORT/charts/false_positives.png', "False Positive Elimination")
    
    # Section 6
    pdf.chapter_title("6", "Jetson Claims & Performance")
    pdf.chapter_body("It is critical to separate development benchmarking from physical edge benchmarking. The reported 16.2 FPS metric was achieved on an RTX development workstation. This does NOT prove target hardware performance.")
    pdf.bullet_list([
        "RTX Development Benchmark: 16.2 FPS (PASS)",
        "Jetson Hardware Benchmark: PENDING"
    ])
    
    # Section 7
    pdf.chapter_title("7", "Deployment Status")
    pdf.chapter_body("The TensorRT/ONNX deployment pipeline status is accurately tracked below to prevent premature claims of hardware validation:")
    pdf.bullet_list([
        "ONNX Export Script: READY",
        "ONNX Artifact Validation: PENDING",
        "TensorRT Instructions: READY",
        "TensorRT Engine Validation: PENDING",
        "Jetson Physical Validation: PENDING"
    ])
    
    # Section 8
    pdf.chapter_title("8", "Violation Engine & Real Evidence")
    pdf.chapter_body("The violation engine correlates spatial overlap with zone rules and temporal validation to ensure high confidence. Real-world validation generated the following evidence frames:")
    pdf.add_image("notebook/outputs/phase3e_positive_validation/evidence/1082af0b.jpg", "Validated Violation Evidence (Helmet/Vest Detection)")
    
    # Section 9
    pdf.chapter_title("9", "Web Application & Rollback")
    pdf.chapter_body("The Next.js dashboard provides visualization. *Note: Data shown in any dashboard screenshots (e.g. 2.8 FPS, 19 violations) are UI demonstration data. Not production benchmark results.*")
    
    pdf.chapter_body("\nRollback Path:")
    pdf.chapter_body("If the V3-HN model encounters regressions, the system uses the preserved fallback model located at:")
    pdf.chapter_body("02_MODELS/rollback/ppe_v2_backup.pt", bold=True)
    
    pdf.output("12_FINAL_REPORT/EDGEVISION_FINAL_REPORT.pdf")

# ---------------------------------------------------------
# 2. MARKDOWN SOURCE GENERATION
# ---------------------------------------------------------
def generate_markdown():
    md = """# EdgeVision PPE Compliance Platform - Final Report Source

## 1. Executive Summary
The EdgeVision PPE Compliance Platform is an edge-deployed computer vision system designed to enforce industrial safety protocols in real-time. It monitors restricted zones, verifies required PPE (helmets and vests), and performs temporal validation to ensure high precision in violation alerting.

This final report details the successful promotion of the V3-HN YOLOv8 model, which utilizes a hard-negative mining strategy to eliminate false positives seen in early prototypes.

## 2. System Architecture
EdgeVision utilizes a modular microservices architecture designed for real-time edge processing.
- Camera -> Person Detection -> Tracking -> PPE Detection -> Zone Engine -> Temporal Validator -> DB.

## 3. Dataset & Hard Negatives
The system is trained on the Construction-PPE dataset.
To resolve false positives (e.g., detecting yellow pipes as helmets), the V3-HN dataset was enriched with visually similar background images (hard negatives) containing no labels.

## 4. Model Development & Architecture
The final production model (V3-HN) architecture exactly matches the validated .pt metadata:
- Base Architecture: YOLOv8n (Nano)
- Layers: 130
- Parameters: 3,012,018
- Gradients: 0
- GFLOPs: 8.2
- Classes: 6 (person, helmet, no_helmet, vest, boots, harness)

**Model Trade-off Decision:**
V3-HN trades a modest reduction in helmet/vest recall for substantially improved real-world false-positive behavior and improved no-helmet recall.

## 5. Model Decision Table

| Metric | V2 | V3-HN | V3-BOOTS |
|---|---|---|---|
| Role | Rollback | Production | Experimental |
| mAP50 | 84.45% | 84.20% | 80.10% |
| Helmet AP50 | 90.15% | 88.65% | 82.20% |
| Vest AP50 | 82.37% | 79.77% | 86.10% |
| Boots AP50 | N/A | N/A | 80.10% |
| Boots Recall | N/A | N/A | 75.30% |
| Warm FPS | 24.3 | 16.2 | 27.82 |
| Helmet FP | 154 | 0 | N/A |
| Vest FP | 10 | 0 | N/A |

**V3-BOOTS is experimental only and is NOT the production model. V3-HN remains production.**

## 6. Jetson Claims & Performance
- RTX Development Benchmark: 16.2 FPS
- Jetson Hardware Benchmark: PENDING
*Note: RTX FPS does not prove target hardware performance.*

## 7. Deployment Status
- ONNX Export Script: READY
- ONNX Artifact Validation: PENDING
- TensorRT Instructions: READY
- TensorRT Engine Validation: PENDING
- Jetson Physical Validation: PENDING

## 8. Web Application & Rollback
*Note: Any Dashboard values (e.g., 2.8 FPS, 19 violations) are UI demonstration data. Not production benchmark results.*

**Rollback Path:** `02_MODELS/rollback/ppe_v2_backup.pt`
"""
    with open("12_FINAL_REPORT/FINAL_REPORT_SOURCE.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    generate_pdf()
    generate_markdown()
    print("PDF and Markdown generated successfully.")
