import os
import sys

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import models, database
from sqlalchemy.orm import Session

def init_db():
    print("Creating PostgreSQL tables...")
    models.Base.metadata.create_all(bind=database.engine)
    print("Tables created successfully.")

def seed_data():
    """Insert default zones and cameras if they don't exist."""
    db: Session = database.SessionLocal()
    try:
        # Seed Zones
        if db.query(models.Zone).count() == 0:
            print("Seeding default zones...")
            zones = [
                models.Zone(
                    name="Construction Zone", type="construction",
                    required_ppe=["helmet", "vest", "boots", "harness"],
                    confidence_threshold=0.25, min_seconds_in_zone=2.0, is_active=True
                ),
                models.Zone(
                    name="Work at Height", type="height",
                    required_ppe=["helmet", "vest", "boots", "harness", "hook"],
                    confidence_threshold=0.25, min_seconds_in_zone=3.0, is_active=True
                ),
                models.Zone(
                    name="General Plant", type="general",
                    required_ppe=["helmet", "vest", "boots", "harness"],
                    confidence_threshold=0.25, min_seconds_in_zone=2.0, is_active=True
                ),
            ]
            db.add_all(zones)
            db.commit()
            print(f"  ✓ Seeded {len(zones)} zones")
        else:
            print(f"  Zones already exist ({db.query(models.Zone).count()} found), skipping.")

        # Seed Camera
        if db.query(models.Camera).count() == 0:
            print("Seeding default camera...")
            # Link to the first zone
            zone = db.query(models.Zone).first()
            camera = models.Camera(
                name="Main Floor (Demo)",
                source_type="LOCAL_VIDEO",
                local_video_path="14_DEMO/test.mp4",
                is_active=True,
                zone_id=zone.id if zone else None
            )
            db.add(camera)
            db.commit()
            print("  ✓ Seeded 1 camera")
        else:
            print(f"  Cameras already exist ({db.query(models.Camera).count()} found), skipping.")

    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    seed_data()
    print("\n✅ Database initialized and seeded successfully!")
