# PPE Extension Dataset Audit

## Findings
- **Boots**: Annotations found in the `construction-ppe` dataset.
- **Harness**: No annotations found in the dataset.

## Statistics

| class | train_images | train_instances | val_images | val_instances | mean_box_area | small_box_percentage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| boots | 538 | 1251 | 64 | 151 | 0.0178 | 39.44 |
| harness | 0 | 0 | 0 | 0 | 0.0 | 0.0 |
| helmet | 840 | 1357 | 107 | 201 | 0.0162 | 44.22 |
| no_helmet | 232 | 400 | 27 | 45 | 0.0270 | 29.44 |
| vest | 837 | 1283 | 109 | 171 | 0.0829 | 10.32 |

## Conclusion
BOOTS DATA: SUFFICIENT
HARNESS DATA: INSUFFICIENT

An additional dataset containing `harness` annotations is REQUIRED before experimental training can begin. Fabrication of labels is forbidden. The experiment must be halted until proper data is acquired.
