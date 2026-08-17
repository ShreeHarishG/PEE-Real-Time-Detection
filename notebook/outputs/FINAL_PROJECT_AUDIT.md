# EdgeVision Final Project Audit

## PROJECT STATUS

**Model:** PASS  
*(V3-HN frozen and preserved. V2 backup preserved. Hard-negative mining successfully eliminated FP.)*

**Pipeline:** PASS  
*(Temporal validation, spatial association, and evidence logging cleanly refactored into `src/pipeline.py` without modifying the validated core logic.)*

**Dashboard:** PASS  
*(Streamlit MVP is polished, reliable, robust to missing files, and clearly displays metrics and evidence.)*

**Documentation:** PASS  
*(README, ARCHITECTURE, MODEL, SETUP, TROUBLESHOOTING, and JETSON_DEPLOYMENT created and standardized.)*

**Dependencies:** PASS  
*(Requirements audited. Virtual environment setup documented.)*

**Local inference:** PASS  
*(Pipeline compiles and runs properly with test video, generating CSV and images.)*

**Jetson preparation:** PASS  
*(ONNX export and TensorRT `trtexec` engine-generation commands meticulously documented.)*

**Jetson hardware validation:** PENDING  
*(As per standard, physical deployment tests must be completed on the exact target hardware before marking this as PASS.)*

**ML retraining:** STOPPED  
*(All model experimentation and fine-tuning has ceased.)*

---

## FINAL PROJECT STRUCTURE
```
EdgeVision/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── model_versions.yaml
├── models/
│   ├── ppe_v3_hn_best.pt
│   ├── ppe_v2_backup.pt
│   ├── yolov8n.pt
│   └── yolo26n.pt
├── src/
│   └── pipeline.py
├── dashboard/
│   └── dashboard.py
├── scripts/
│   ├── run_pipeline.ps1
│   ├── run_dashboard.ps1
│   └── [historical phase/audit scripts preserved for reproducibility]
├── docs/
│   ├── ARCHITECTURE.md
│   ├── JETSON_DEPLOYMENT.md
│   ├── SETUP.md
│   ├── MODEL.md
│   └── TROUBLESHOOTING.md
├── datasets/
│   └── [preserved hard-negative and core datasets]
├── outputs/
│   ├── evidence/
│   ├── reports/
│   └── results/
└── [misc cached historical runs preserved for reference]
```

---

## KNOWN LIMITATIONS
1. **Absolute Recall Regression:** The `V3-HN` model sacrifices a small amount of absolute validation recall (84% -> 82%) to entirely suppress real-world false positives. Extremely distant or highly occluded PPE may occasionally be missed in a single frame.
2. **CPU Inference Bottleneck:** The pipeline currently achieves ~16.2 FPS on standard hardware. Achieving the 12+ FPS floor consistently on edge hardware will heavily depend on successfully compiling the `.engine` TensorRT file.
3. **Tracking Gaps:** ByteTrack is reliant on the `yolov8n` person detector. If the person detector drops a worker for extended frames, the temporal validator might reset its history for that worker.

---

## REMAINING ACTIONS
1. **Physical Jetson Validation:** Procure the target Jetson Orin device, convert the model to TensorRT via `trtexec`, and run a continuous 8-hour benchmark to verify thermal limits and FPS.
2. **PostgreSQL Integration:** If long-term persistence is requested later, replace the CSV logging in `src/pipeline.py` with SQLAlchemy ORM hooks (schema details were provided in the original PRD).
3. **DeepStream Port (Optional):** If Python overhead causes the Jetson to struggle, the pipeline logic (TemporalValidator & Association) will need to be written as a C++ GStreamer plugin for DeepStream.
