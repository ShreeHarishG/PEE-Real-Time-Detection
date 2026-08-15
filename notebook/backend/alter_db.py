from sqlalchemy import create_engine, text
import os

DATABASE_URL = "postgresql://edgevision:edgevision_password@localhost:5432/edgevision"
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE processing_jobs ADD COLUMN workers_detected INTEGER DEFAULT 0;"))
        conn.execute(text("ALTER TABLE processing_jobs ADD COLUMN violations_detected INTEGER DEFAULT 0;"))
        conn.execute(text("ALTER TABLE processing_jobs ADD COLUMN output_video_path VARCHAR;"))
        conn.commit()
    except Exception as e:
        print("Processing jobs error:", e)
        
    try:
        conn.execute(text("ALTER TABLE violation_events ADD COLUMN job_id VARCHAR REFERENCES processing_jobs(id);"))
        conn.execute(text("ALTER TABLE violation_events ADD COLUMN video_timestamp_sec FLOAT;"))
        conn.commit()
    except Exception as e:
        print("Violation events error:", e)
        
    print("DB Alteration script done.")
