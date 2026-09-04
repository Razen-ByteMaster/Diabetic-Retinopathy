# 👁️ Diabetic Retinopathy Dataset

> 2,750 retina fundus photos across 5 severity grades — ready to train a classifier on.
> Curated data, documented classes, zero guesswork. Just add a model. 🔬

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ready-orange?logo=pytorch)
![Images](https://img.shields.io/badge/Images-2750-green)
![Classes](https://img.shields.io/badge/Classes-5-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

## ✨ What is this?

A local retinal-image dataset for diabetic retinopathy (DR) grading — the classic 5-level severity scale used in screening. Each folder is one grade, each file one fundus photo. Built to train image classifiers (like the ResNet50s in the Malaria/Pneumonia projects).

| Grade | Folder | Images | Share |
|---|---|---|---|
| 0 — Healthy | `Healthy` | 1000 | 36% |
| 1 — Mild DR | `Mild DR` | 370 | 13% |
| 2 — Moderate DR | `Moderate DR` | 900 | 33% |
| 3 — Severe DR | `Severe DR` | 190 | 7% |
| 4 — Proliferative DR | `Proliferate DR` | 290 | 11% |
| **Total** | | **2750** | ~350 MB |

> ⚠️ **Imbalanced by nature** (Severe = 7% vs Healthy = 36%) — use stratified splits and class-weighted loss, or the model will just learn to say "healthy."

## 🚀 Quickstart

```bash
pip install -r requirements.txt
```

Load it straight into PyTorch — folder names become labels automatically:

```python
from torchvision import datasets, transforms

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
data = datasets.ImageFolder(root=".", transform=tf)
print(data.classes)   # ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']
print(len(data))      # 2750
```

> ⚠️ **The 350 MB of PNGs are NOT in git** (see `.gitignore`). Clone the repo, then drop the image folders next to this README — same 5 folder names — and the snippet above works.

## 🧬 Source

Grading follows the APTOS 2019 blindness-detection scale (0–4):

- Competition: https://www.kaggle.com/competitions/aptos2019-blindness-detection

## 🛠️ Suggested stack

| Step | Tech |
|---|---|
| Model | ResNet50 / EfficientNet (torchvision) |
| Training | PyTorch + weighted CrossEntropy + AdamW |
| Splits | Stratified train/val/test (e.g. 70/15/15) |
| Metrics | Accuracy + per-class F1 (accuracy alone lies on imbalanced data) |
| Explainability | Grad-CAM (show *where* the lesions are) |

## 📁 Project structure

```
Diabetic-Retinopathy/
├── Healthy/             # 1000 images (local only, gitignored)
├── Mild DR/             # 370 images (local only, gitignored)
├── Moderate DR/         # 900 images (local only, gitignored)
├── Severe DR/           # 190 images (local only, gitignored)
├── Proliferate DR/      # 290 images (local only, gitignored)
├── requirements.txt     # Python dependencies
└── README.md            # You are here 👋
```

## 🗺️ Roadmap

- [ ] Train/test split script (`split.py`, stratified)
- [ ] Baseline ResNet50 training notebook
- [ ] Per-class F1 + confusion matrix report
- [ ] Grad-CAM lesion heatmaps
- [ ] Flask/FastAPI serving endpoint (match the Malaria/Pneumonia style)

## 📄 License

MIT. Built by [Razen Moamen](https://github.com/Razen-ByteMaster) as a portfolio project. PRs welcome! 🎉

> ⚕️ **Disclaimer:** educational/portfolio project — not a medical device. Never use it for real diagnosis.
