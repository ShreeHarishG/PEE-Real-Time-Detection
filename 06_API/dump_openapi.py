import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "notebook", "backend")))

try:
    from app.main import app
    schema = app.openapi()
    with open("w:/3 projects/Building/Tfrenzy/06_API/openapi.json", "w") as f:
        json.dump(schema, f, indent=2)
    print("OpenAPI schema dumped successfully.")
except Exception as e:
    print(f"Failed to dump schema: {e}")
