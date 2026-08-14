# Project Handover

**PROJECT:** EdgeVision PPE Compliance and Work-at-Height Safety Platform
**PRODUCTION MODEL:** V3-HN (`ppe_v3_hn_best.pt`)
**ROLLBACK MODEL:** V2 (`ppe_v2_backup.pt`)
**EXPERIMENTAL MODEL:** V3-BOOTS (`ppe_v3_boots_best.pt`)

**VALIDATED PPE:**
- Helmet
- Vest

**UNSUPPORTED/PENDING PPE:**
- Boots (Bounding box inconsistency during V3-BOOTS testing)
- Harness (Not enough high-quality data)
- Lanyard
- Hook
- Anchor Point

**DEVELOPMENT HARDWARE:** RTX GPU Workstation
**TARGET HARDWARE:** Jetson Orin Nano / NX

**JETSON VALIDATION:** PENDING
**DEEPSTREAM VALIDATION:** PENDING
**ONNX:** PENDING EXPORT (Script provided in `07_DEPLOYMENT/scripts/export_onnx.py`)
**TENSORRT:** REPRODUCIBLE (Instructions in `07_DEPLOYMENT/TENSORRT.md`)
