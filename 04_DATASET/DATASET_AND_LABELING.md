# Dataset and Labeling Documentation

## 1. Dataset Overview

The EdgeVision V3-HN production model utilizes a heavily curated dataset optimized for zero-false-positive detection of core safety gear.

**Primary Dataset:** Construction-PPE
**Location:** `../notebook/datasets/construction-ppe`

*Note: The physical dataset images (multi-GB) are retained in the `notebook/datasets/` folder to prevent redundant file duplication in this submission package.*

## 2. Supported Classes (V3-HN)

The production model is explicitly trained and validated for the following classes:

- `0`: `person`
- `1`: `helmet`
- `2`: `no_helmet`
- `3`: `vest`

## 3. Unsupported / Pending Classes

The following classes were evaluated during the V3-BOOTS experimental phase but were rejected due to insufficient dataset quality (small object bounding box inconsistencies):

- `boots`
- `harness`
- `lanyard`
- `hook`
- `anchor_point`

These classes are currently marked as **UNTRAINED** in the UI to prevent any misrepresentation of system capabilities.

## 4. Hard-Negative Mining Strategy

To achieve zero false positives during Phase 3 of the project audit, a hard-negative mining strategy was implemented:

1. Visually similar objects (e.g., specific colored clothing misclassified as vests) were collected.
2. These objects were added to the training set as empty background images (no bounding boxes).
3. This process successfully suppressed false positives, sacrificing a nominal amount of absolute recall (84% -> 82%) to ensure 100% precision in real-world violation events.
