# Database Documentation

## Architecture

EdgeVision V3 utilizes **PostgreSQL** as the primary relational database. 
All obsolete SQLite artifacts from the earlier generic V1 implementations have been completely purged from the production pipeline.

## Configuration

The database connection is managed via standard SQLAlchemy configurations located in `../notebook/backend/app/database.py`. 

The connection string defaults to:
`postgresql://edgevision:<password>@localhost:5432/edgevision`

*Note: For the final submission, no hardcoded secrets or production passwords are included in the repository. Please refer to `.env.example` in the `notebook/` folder for required environment variables.*

## Schema and Initialization

The database schema is defined using SQLAlchemy ORM models located in `../notebook/backend/app/models.py`.

To initialize the schema on a fresh PostgreSQL instance, run:
```bash
python 05_DATABASE/init_db.py
```
*(Alternatively, you can run the original script directly at `notebook/backend/scripts/init_db.py`)*

The `init_db.py` script will automatically map the ORM models to the PostgreSQL instance and create the required tables for:
- Cameras
- Zones (and required PPE configurations)
- Violations (and associated temporal evidence paths)
- Processing Jobs
