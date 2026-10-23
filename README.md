# CNN-Based Multiclass Classification of Hot-Rolled Steel Surface Defects
Harsitha G P | RA2311026050197 | IV AI & ML - A

Dataset: NEU Surface Defect Database (NEU-CLS), 1,800 grayscale images, 6 classes.

## Setup
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt
Place the dataset anywhere and point to it with `--data_root`, the env var `NEU_DATASET_ROOT`, or `src/config.py`.

## Stage status
- [x] Stage 1 - environment check + dataset inspection + EDA (`python -m src.eda`)
- [ ] Stage 2 - preprocessing + stratified split
- [ ] Stage 3 - augmentation
- [ ] Stage 4 - models
- [ ] Stage 5 - training
- [ ] Stage 6 - evaluation + confusion analysis
- [ ] Stage 7 - Grad-CAM
- [ ] Stage 8 - robustness + ablation
- [ ] Stage 9 - final report (fill outputs/reports/PROJECT_REPORT_TEMPLATE.md with REAL results)

Run all commands from the project root (the folder containing `src/`).
