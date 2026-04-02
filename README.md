# ⚽ A Computer Vision Pipeline for Football Table Analytics and Game Insights

End-to-end foosball game analysis and winner prediction using a
single fixed overhead webcam — no additional sensors or manual
intervention required.

> **Master's Thesis** — University of Siegen, Computer Vision Group
> Aakash Deshpande | Supervised by Alexander Aurus |
> Prof. Dr. Michael Möller

---

## 📌 Overview

The pipeline takes a 60-second MP4 video as input and predicts
**White**, **Black**, or **Draw** as the winner.
```
Raw MP4 → Calibration → Perspective Correction
       → Ball Tracking → Player Detection
       → Branch 1: Hit Detection
       → Branch 2: TCN Winner Prediction
```

---

## 📊 Results

Evaluated across 5 random seeds:

| Metric | Mean | Std |
|--------|------|-----|
| Sequence-level accuracy | 0.487 | 0.017 |
| Hard vote accuracy | 0.575 | 0.100 |
| Soft vote accuracy | 0.525 | 0.050 |

- Uniform random baseline: **0.333**
- Majority-class baseline: **0.375**
- **3184 hits** detected across 41 non-Draw videos

---

## 🗃 Repository Structure

| Folder | Contents |
|--------|----------|
| `1_calibration/` | Camera calibration pipeline |
| `2_tracking/` | Ball, player, and hit detection |
| `3_post_processing/` | Dataset construction |
| `4_model/` | TCN training and evaluation |
| `thesis/` | Full Master's thesis PDF |

---

## ⚙️ Installation
```bash
git clone https://github.com/AakashDeshpande97/A-Computer-Vision-Pipeline-for-Football-Table-Analytics-and-Game-Insights.git
cd A-Computer-Vision-Pipeline-for-Football-Table-Analytics-and-Game-Insights
pip install -r requirements.txt
```

---

## ▶️ Usage

Run notebooks in order:
```
1_calibration → 2_tracking → 3_post_processing → 4_model
```

> Calibration artefacts (map1.npy, map2.npy, newK.npy) and
> video files are not included. Run calibration notebook first.

---

## 🙏 Acknowledgements

Supervised by **Alexander Aurus** and
**Prof. Dr. Michael Möller**,
Computer Vision Group, University of Siegen.

---

## 📜 Licence

MIT Licence — see [LICENSE](LICENSE) for details.