from sqlalchemy import create_engine, text
engine = create_engine('postgresql://edgevision:edgevision_password@localhost:5432/edgevision')
with engine.connect() as conn:
    conn.execute(text('ALTER TABLE violation_events ADD COLUMN feedback_boots BOOLEAN NULL;'))
    conn.commit()
print('Success!')
