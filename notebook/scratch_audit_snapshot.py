import os
import hashlib
import json
import glob

def get_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    root_dir = r"w:\3 projects\Building\Tfrenzy"
    manifest = {
        "root_dir": root_dir,
        "models": {},
        "files": [],
        "configurations": {}
    }

    # Model Hashes
    models_dir = os.path.join(root_dir, "notebook", "models")
    models_to_hash = ["ppe_v3_hn_best.pt", "ppe_v2_backup.pt", "ppe_v3_boots_best.pt"]
    
    for m in models_to_hash:
        path = os.path.join(models_dir, m)
        manifest["models"][m] = {
            "path": path,
            "hash": get_sha256(path)
        }

    # Get all files
    for root, dirs, files in os.walk(root_dir):
        if "node_modules" in root or ".git" in root or ".next" in root or "__pycache__" in root:
            continue
        for f in files:
            manifest["files"].append(os.path.join(root, f))
            
    # Config files of interest
    config_paths = [
        os.path.join(root_dir, "notebook", "config", "model_versions.yaml"),
        os.path.join(root_dir, "notebook", "backend", "app", "database.py"),
        os.path.join(root_dir, "notebook", "frontend", "next.config.ts"),
        os.path.join(root_dir, "notebook", "docker-compose.yml"),
        os.path.join(root_dir, "notebook", ".env.example")
    ]
    for cp in config_paths:
        if os.path.exists(cp):
            manifest["configurations"][os.path.basename(cp)] = cp

    output_file = os.path.join(root_dir, "_submission_backup_manifest.json")
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest written to {output_file}")

if __name__ == "__main__":
    main()
