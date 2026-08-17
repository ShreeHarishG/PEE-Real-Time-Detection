# Harness Dataset Audit

## Discovery Results
During the investigation phase, several dataset sources were examined:
1. **Roboflow Safety Harness Detection dataset**: Highly relevant, but direct dataset download requires manual API authentication or UI interaction. Not available for direct unauthenticated script downloading.
2. **"A novel computer vision-based approach for monitoring safety harness use in construction"**: As noted in the paper's data availability statement, this dataset is restricted and requires a direct request to the corresponding author. It is not publicly hosted.
3. **SH17 Dataset (GitHub)**: Downloaded and audited. While it claims to have 17 PPE classes, it does not include `safety_harness` (only safety-suit and safety-vest).

## Decision
Because we cannot automatically download a verified, legitimate dataset containing actual `safety_harness` bounding boxes, we must classify this discovery as **UNUSABLE**. Fabricating labels or misclassifying `safety-suit` as a harness is strictly prohibited.

## Statistics

| class | train_images | train_instances | val_images | val_instances | test_images | test_instances | mean_box_area | small_box_percentage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| safety_harness | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
