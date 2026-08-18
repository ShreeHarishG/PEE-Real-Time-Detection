import os
from sqlalchemy import create_engine, text

# Use the environment variable if present, otherwise default to the one from .env.example
database_url = os.environ.get("DATABASE_URL", "postgresql://edgevision:edgevision_password@localhost:5432/edgevision")

engine = create_engine(database_url)

with engine.connect() as conn:
    conn.execute(text('ALTER TABLE violation_events ADD COLUMN feedback_harness BOOLEAN NULL;'))
    conn.commit()

print('Migration successful: Added feedback_harness to violation_events')
