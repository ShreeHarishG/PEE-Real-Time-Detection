import os
import re

def main():
    root_dir = r"w:\3 projects\Building\Tfrenzy"
    output_file = os.path.join(root_dir, "notebook", "outputs", "pre_restructure_audit.md")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    hardcoded_w = []
    
    for root, dirs, files in os.walk(root_dir):
        if "node_modules" in root or ".git" in root or ".next" in root or "__pycache__" in root:
            continue
        for f in files:
            if not (f.endswith(".py") or f.endswith(".ts") or f.endswith(".tsx") or f.endswith(".js") or f.endswith(".md") or f.endswith(".txt") or f.endswith(".yaml") or f.endswith(".yml")):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    for i, line in enumerate(file):
                        if "W:\\" in line.upper():
                            hardcoded_w.append(f"{path}:{i+1}: {line.strip()}")
            except Exception:
                pass

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Pre-Restructure Audit\n\n")
        f.write("## Actual roots\n")
        f.write(f"- Root: {root_dir}\n")
        f.write("- Working App Root: notebook\n")
        f.write("- Frontend Root: notebook/frontend\n")
        f.write("- Backend Root: notebook/backend\n")
        f.write("- ML Pipeline Root: notebook/src\n")
        f.write("- Model Dir: notebook/models\n")
        f.write("- Dataset Dir: notebook/datasets\n")
        f.write("- Deployment Dir: notebook/deployment\n")
        f.write("- Documentation Dir: docs and notebook/outputs\n\n")
        f.write("## Hardcoded W:\\ Paths Found\n")
        if hardcoded_w:
            for item in hardcoded_w:
                f.write(f"- {item}\n")
        else:
            f.write("- None found\n")
            
    print(f"Audit written to {output_file}")

if __name__ == "__main__":
    main()
