import os
import sys

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import models, database

def init_db():
    print("Creating PostgreSQL tables...")
    models.Base.metadata.create_all(bind=database.engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    init_db()
