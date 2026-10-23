# CNN-Based Multiclass Classification of Hot-Rolled Steel Surface Defects
**Harsitha G P - RA2311026050197 - IV AI & ML "A"**

> Placeholders marked `[TO FILL]` are completed only from experiments actually run.
> No numbers are included until then.

## 1. Project Overview
[TO FILL: short summary]
## 2. Problem Statement
Classify grayscale hot-rolled steel strip images into six surface-defect categories (rolled-in scale, patches, crazing, pitted surface, inclusion, scratches) for automated quality control, using only 300 images per class, with high intra-class variation and inter-class similarity.
## 3. Objectives
Design, train and evaluate CNN models; compare a custom CNN with transfer learning; evaluate with accuracy, precision, recall, F1 and confusion matrix; analyse class confusion; apply Grad-CAM; study augmentation and architecture effects; select the best model on overall and per-class performance.
## 4. Dataset Description
[TO FILL from outputs/reports/dataset_stats.json and figures 01-05]
## 5. Methodology
5.1 Preprocessing and stratified split [Stage 2]  
5.2 Augmentation and justification [Stage 3]  
5.3 Training procedure [Stage 5]
## 6. Architecture
6.1 Custom CNN [Stage 4]  6.2 ResNet18 / EfficientNet-B0: frozen vs fine-tuned [Stage 4]
## 7. Experimental Setup
[TO FILL from outputs/reports/environment.json and saved experiment configs]
## 8. Results
8.1 Model comparison table [TO FILL]  8.2 Augmentation comparison [TO FILL]  8.3 Ablation [TO FILL]  8.4 Per-class metrics and ROC-AUC [TO FILL]  8.5 Confusion matrix [TO FILL]  8.6 Grad-CAM [TO FILL]  8.7 Robustness [TO FILL]
## 9. Discussion
[TO FILL: only conclusions supported by results; confusion pairs from the actual confusion matrix]
## 10. Limitations
Small dataset (1,800 images); single dataset/source; results from a single split/seed unless repeated; Grad-CAM is a visualisation of contributing regions, not proof of causal reasoning. [TO FILL: add observed limitations]
## 11. Future Work
[TO FILL]
## 12. Conclusion
[TO FILL]
