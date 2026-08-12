import os
import pandas as pd
from collections import Counter

CSV_PATH = "edgevision_v2/outputs/logs/violations_functional.csv"
OUTPUT_CSV = "outputs/phase3c_violation_audit.csv"
OUTPUT_HTML = "outputs/phase3c_evidence_audit.html"

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV not found at {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
df = df.fillna("")

audit_records = []
stats = {
    "total_events": len(df),
    "evidence_found": 0,
    "missing_evidence": 0,
    "unsupported_ppe": 0,
    "empty_missing_ppe": 0
}

duplicate_ids = [k for k, v in Counter(df['event_id']).items() if v > 1]
unique_persons = set(df['person_id'])

# The model only supports these:
SUPPORTED_PPE = {"helmet", "vest", "no_helmet"}

html_parts = [
    "<html><head><style>",
    "body { font-family: sans-serif; background: #1e1e1e; color: #fff; padding: 20px; }",
    ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }",
    ".card { background: #2d2d2d; border-radius: 8px; padding: 15px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }",
    ".card img { width: 100%; border-radius: 4px; margin-bottom: 10px; }",
    ".field { margin: 5px 0; font-size: 14px; }",
    ".label { color: #888; font-weight: bold; }",
    "</style></head><body>",
    "<h1>Phase 3C Violation Audit</h1>",
    "<div class='grid'>"
]

for idx, row in df.iterrows():
    # Attempt to resolve the evidence path
    ev_path = row['evidence_path']
    if not os.path.exists(ev_path):
        # Fallback to edgevision_v2 directory
        ev_path = os.path.join("edgevision_v2", ev_path)
    
    file_exists = os.path.exists(ev_path)
    if file_exists:
        stats['evidence_found'] += 1
    else:
        stats['missing_evidence'] += 1
        
    missing_ppe = str(row['missing_ppe']).strip()
    if not missing_ppe:
        stats['empty_missing_ppe'] += 1
    
    ppe_list = [p.strip() for p in missing_ppe.strip("[]'\"").split(",") if p.strip()]
    unsupported = [p for p in ppe_list if p not in SUPPORTED_PPE and p != ""]
    if unsupported:
        stats['unsupported_ppe'] += 1
        
    audit_records.append({
        "event_id": row['event_id'],
        "person_id": row['person_id'],
        "timestamp": row['timestamp'],
        "zone": row['zone'],
        "missing_ppe": missing_ppe,
        "confidence": row['confidence'],
        "evidence_path": ev_path,
        "evidence_exists": file_exists
    })
    
    # HTML Card
    abs_img_path = os.path.abspath(ev_path) if file_exists else ""
    # HTML handles local files via file:/// protocol in most browsers
    img_src = f"file:///{abs_img_path}".replace("\\", "/") if file_exists else ""
    img_tag = f"<img src='{img_src}' alt='Evidence Image'>" if file_exists else "<div style='color:red; padding: 50px 0; text-align:center;'>Image Not Found</div>"
    
    html_parts.append(f"""
        <div class="card">
            {img_tag}
            <div class="field"><span class="label">Event ID:</span> {row['event_id']}</div>
            <div class="field"><span class="label">Person ID:</span> {row['person_id']}</div>
            <div class="field"><span class="label">Timestamp:</span> {row['timestamp']}</div>
            <div class="field"><span class="label">Zone:</span> {row['zone']}</div>
            <div class="field"><span class="label">Missing PPE:</span> {missing_ppe}</div>
            <div class="field"><span class="label">Confidence:</span> {row['confidence']}</div>
        </div>
    """)

html_parts.append("</div></body></html>")

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(OUTPUT_CSV, index=False)

print("\n=== PHASE 3C AUDIT REPORT ===")
print(audit_df[['event_id', 'person_id', 'missing_ppe', 'evidence_exists']].to_string(index=False))

print("\n=== SUMMARY STATISTICS ===")
print(f"Total Events: {stats['total_events']}")
print(f"Evidence Files Found: {stats['evidence_found']}")
print(f"Missing Evidence Files: {stats['missing_evidence']}")
print(f"Duplicate Event IDs: {len(duplicate_ids)}")
print(f"Empty Missing_PPE Fields: {stats['empty_missing_ppe']}")
print(f"Unsupported PPE Classes: {stats['unsupported_ppe']}")
print(f"Unique Person IDs: {len(unique_persons)}")
print("\nFiles generated:")
print(f"- {OUTPUT_CSV}")
print(f"- {OUTPUT_HTML} (Open this in your browser to inspect images)")
