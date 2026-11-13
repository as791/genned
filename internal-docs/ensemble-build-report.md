## Model build parity

40 dataset images, app normalization. A logit-gap difference below ~0.05 is negligible after calibration (slopes are ~0.2).

| Check | Max \|gap diff\| | Mean | OK |
|---|---|---|---|
| commfor-224: PyTorch vs ONNX export, same input | 0.0000 | 0.0000 | yes |
| commfor-224: authors' preprocessing + PyTorch vs app preprocessing + ONNX | 0.0000 | 0.0000 | yes |
| bundled: fp32 vs fp16 weights | 0.0132 | 0.0045 | yes |
| commfor-224: fp32 vs fp16 weights | 0.0100 | 0.0033 | yes |
| ai-image-detector.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |
| commfor-224.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |

| File | fp32 MB | Shipped (fp16 weights) MB |
|---|---|---|
| ai-image-detector.onnx | 70.1 | 35.2 |
| commfor-224.onnx | 86.8 | 43.5 |
# App ensemble: bundled + Community Forensics 224

Scores come from the exact ONNX files the app bundles (fp16 weights). The ensemble score is the mean of both models' standardized logits; standardization is fit on the photo scores and shared with video: mean_d=-0.0905, std_d=3.4163, mean_c=-1.4826, std_c=6.0167.

### Photos: each model alone vs the ensemble

| Score | AUC defactify | AUC mj-dalle-sd-nbp | AI caught @5% FA defactify | AI caught @5% FA mj-dalle-sd-nbp |
|---|---|---|---|---|
| Bundled (two views) | 0.957 | 0.855 | 78.1% | 49.5% |
| Community Forensics 224 | 0.986 | 0.794 | 94.4% | 36.3% |
| **Ensemble** | 0.997 | 0.873 | 97.6% | 52.7% |

## Preprocess: `ensemble (photos)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| defactify | 1500 | 0.997 | 0.261 | 0.078 | 0.044 | 0.0% | 28.1% |
| mj-dalle-sd-nbp | 1500 | 0.873 | 0.163 | 0.105 | 0.055 | 0.0% | 9.9% |

Pooled fit: **slope = 3.9437, intercept = 0.2725**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 79.2% | 2.8% |
| 0.10 | 66.0% | 5.6% |
| 0.15 | 54.8% | 8.0% |
| 0.20 | 47.6% | 12.4% |
| 0.25 | 42.8% | 13.2% |
| 0.30 | 36.8% | 15.2% |
| 0.35 | 32.0% | 17.2% |
| 0.40 | 27.6% | 19.2% |
| 0.45 | 23.6% | 22.8% |
| 0.50 | 19.6% | 24.4% |
| 0.55 | 17.2% | 26.4% |
| 0.60 | 14.0% | 30.0% |
| 0.65 | 13.2% | 35.2% |
| 0.70 | 11.2% | 39.2% |
| 0.75 | 10.4% | 42.4% |
| 0.80 | 6.4% | 46.0% |
| 0.85 | 4.4% | 50.4% |
| 0.90 | 3.2% | 58.0% |
| 0.95 | 2.0% | 64.8% |

Recommendation: **HIGH ≥ 0.85**, **LOW < 0.15** → worst case 4.4% of real images shown HIGH (target ≤5%), 8.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.20), worst case: **3.2% of real shown HIGH**, 12.4% of AI shown LOW, and at least 42.0% of AI shown HIGH.

### Video: each model alone vs the ensemble

| Score | AUC deepaction | AUC df26 | AI caught @5% FA deepaction | AI caught @5% FA df26 |
|---|---|---|---|---|
| Bundled (two views) | 0.731 | 0.764 | 26.0% | 20.0% |
| Community Forensics 224 | 0.868 | 0.826 | 53.0% | 57.0% |
| **Ensemble** | 0.865 | 0.829 | 43.0% | 49.0% |

## Preprocess: `ensemble (video)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| deepaction | 200 | 0.865 | 0.162 | 0.146 | 0.089 | 0.0% | 1.5% |
| df26 | 200 | 0.829 | 0.198 | 0.163 | 0.093 | 0.0% | 0.5% |

Pooled fit: **slope = 2.8736, intercept = -1.1606**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 99.0% | 0.0% |
| 0.10 | 98.0% | 0.0% |
| 0.15 | 92.0% | 3.0% |
| 0.20 | 83.0% | 6.0% |
| 0.25 | 73.0% | 10.0% |
| 0.30 | 67.0% | 18.0% |
| 0.35 | 55.0% | 21.0% |
| 0.40 | 43.0% | 27.0% |
| 0.45 | 38.0% | 29.0% |
| 0.50 | 32.0% | 37.0% |
| 0.55 | 28.0% | 41.0% |
| 0.60 | 24.0% | 48.0% |
| 0.65 | 21.0% | 54.0% |
| 0.70 | 12.0% | 60.0% |
| 0.75 | 11.0% | 66.0% |
| 0.80 | 6.0% | 68.0% |
| 0.85 | 5.0% | 79.0% |
| 0.90 | 1.0% | 86.0% |
| 0.95 | 0.0% | 90.0% |

Recommendation: **HIGH ≥ 0.85**, **LOW < 0.25** → worst case 5.0% of real images shown HIGH (target ≤5%), 10.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **1.0% of real shown HIGH**, 3.0% of AI shown LOW, and at least 14.0% of AI shown HIGH.

# Genned detector accuracy benchmark

Model: bundled `Dafilab/ai-image-detector` ONNX export. `squash` = what the app does today; `social` = long edge ≤1080 + JPEG q75 (Instagram re-upload). Every condition then gets the app's own normalization (long edge ≤2048, JPEG q92), as on-device. Threshold 0.5 for accuracy/FPR/FNR; bands use the app's LOW <20% / HIGH ≥90% (meaningful for calibrated rows such as 'app as shipped').

> If the model was trained on one of these datasets, its numbers there are inflated. Treat a dataset that scores far better than the others with suspicion.

## Overall

| Dataset | Model | Condition | Preprocess | n (AI/real) | AUC | Accuracy | FPR (real→AI) | FNR (AI missed) | Real shown HIGH | AI shown LOW | ECE | Scores >99% or <1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepaction | app as shipped | video | video5 | 100/100 | 0.731 | 52.0% | 0.0% | 96.0% | 0.0% | 1.0% | 0.249 | 0.0% |
| deepaction | commfor-224 (app onnx) | video | video5 | 100/100 | 0.868 | 77.0% | 12.0% | 34.0% | 1.0% | 20.0% | 0.135 | 31.5% |
| defactify | app as shipped | original | avg | 250/250 | 0.957 | 86.6% | 8.0% | 18.8% | 0.0% | 0.0% | 0.208 | 0.0% |
| defactify | app as shipped | jpeg75 | avg | 250/250 | 0.957 | 86.8% | 8.0% | 18.4% | 0.0% | 0.0% | 0.209 | 0.0% |
| defactify | app as shipped | social | avg | 250/250 | 0.957 | 86.8% | 8.0% | 18.4% | 0.0% | 0.0% | 0.209 | 0.0% |
| defactify | commfor-224 (app onnx) | original | native | 250/250 | 0.986 | 92.8% | 2.0% | 12.4% | 0.0% | 6.4% | 0.057 | 70.0% |
| defactify | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.986 | 92.6% | 1.6% | 13.2% | 0.0% | 6.4% | 0.058 | 70.2% |
| defactify | commfor-224 (app onnx) | social | native | 250/250 | 0.986 | 92.6% | 1.6% | 13.2% | 0.0% | 6.4% | 0.058 | 70.2% |
| df26 | app as shipped | video | video5 | 100/100 | 0.764 | 54.0% | 0.0% | 92.0% | 0.0% | 0.0% | 0.192 | 0.0% |
| df26 | commfor-224 (app onnx) | video | video5 | 100/100 | 0.826 | 73.0% | 26.0% | 28.0% | 2.0% | 15.0% | 0.140 | 25.5% |
| mj-dalle-sd-nbp | app as shipped | original | avg | 250/250 | 0.874 | 80.2% | 24.4% | 15.2% | 0.0% | 0.0% | 0.175 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | jpeg75 | avg | 250/250 | 0.837 | 76.6% | 24.0% | 22.8% | 0.0% | 0.0% | 0.143 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | social | avg | 250/250 | 0.855 | 76.4% | 29.2% | 18.0% | 0.0% | 0.0% | 0.140 | 0.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | original | native | 250/250 | 0.812 | 70.4% | 10.0% | 49.2% | 4.0% | 38.8% | 0.213 | 44.8% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.794 | 68.2% | 10.0% | 53.6% | 4.4% | 42.0% | 0.246 | 45.2% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | social | native | 250/250 | 0.775 | 66.6% | 12.0% | 54.8% | 4.0% | 41.6% | 0.244 | 43.6% |

## Per generator (bundled: squash; candidates: native; app ensemble: as shipped)

Share classified correctly at 0.5 (for `real`: share *not* flagged as AI).

### deepaction · app as shipped

| Generator | video |
|---|---|
| real | 100.0% (n=100) |
| BDAnimateDiffLightning | 23.5% (n=17) |
| CogVideoX5B | 0.0% (n=17) |
| RunwayML | 0.0% (n=17) |
| StableDiffusion | 0.0% (n=16) |
| Veo | 0.0% (n=17) |
| VideoPoet | 0.0% (n=16) |

### deepaction · commfor-224 (app onnx)

| Generator | video |
|---|---|
| real | 88.0% (n=100) |
| BDAnimateDiffLightning | 100.0% (n=17) |
| CogVideoX5B | 41.2% (n=17) |
| RunwayML | 47.1% (n=17) |
| StableDiffusion | 93.8% (n=16) |
| Veo | 47.1% (n=17) |
| VideoPoet | 68.8% (n=16) |

### defactify · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 98.0% (n=250) | 98.4% (n=250) | 98.4% (n=250) |
| dalle3 | 92.0% (n=50) | 90.0% (n=50) | 90.0% (n=50) |
| midjourney6 | 58.0% (n=50) | 58.0% (n=50) | 58.0% (n=50) |
| sd21 | 98.0% (n=50) | 98.0% (n=50) | 98.0% (n=50) |
| sd3 | 94.0% (n=50) | 92.0% (n=50) | 92.0% (n=50) |
| sdxl | 96.0% (n=50) | 96.0% (n=50) | 96.0% (n=50) |

### df26 · app as shipped

| Generator | video |
|---|---|
| real | 100.0% (n=100) |
| Grok_Imagine | 40.0% (n=10) |
| HunyuanVideo_1.5_14b_i2v | 10.0% (n=10) |
| HunyuanVideo_1.5_14b_t2v | 10.0% (n=10) |
| Kling_3.0 | 10.0% (n=10) |
| LTX_2.3_distilled_i2v | 0.0% (n=10) |
| LTX_2.3_distilled_t2v | 0.0% (n=10) |
| Veo_3.1 | 0.0% (n=10) |
| Wan_2.2_14b_i2v | 0.0% (n=10) |
| Wan_2.2_14b_t2v | 0.0% (n=10) |
| Wan_2.6 | 10.0% (n=10) |

### df26 · commfor-224 (app onnx)

| Generator | video |
|---|---|
| real | 74.0% (n=100) |
| Grok_Imagine | 100.0% (n=10) |
| HunyuanVideo_1.5_14b_i2v | 50.0% (n=10) |
| HunyuanVideo_1.5_14b_t2v | 50.0% (n=10) |
| Kling_3.0 | 100.0% (n=10) |
| LTX_2.3_distilled_i2v | 20.0% (n=10) |
| LTX_2.3_distilled_t2v | 70.0% (n=10) |
| Veo_3.1 | 100.0% (n=10) |
| Wan_2.2_14b_i2v | 40.0% (n=10) |
| Wan_2.2_14b_t2v | 90.0% (n=10) |
| Wan_2.6 | 100.0% (n=10) |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 90.0% (n=250) | 90.0% (n=250) | 88.0% (n=250) |
| ai | 50.8% (n=250) | 46.4% (n=250) | 45.2% (n=250) |

## Calibration (original, squash)

When the model says X%, how often is the image actually AI?

### deepaction · app as shipped

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 1 | 9.6% | 0.0% |
| 0.1-0.2 | 57 | 16.9% | 24.6% |
| 0.2-0.3 | 91 | 24.1% | 52.7% |
| 0.3-0.4 | 41 | 33.8% | 68.3% |
| 0.4-0.5 | 6 | 43.4% | 100.0% |
| 0.5-0.6 | 3 | 54.2% | 100.0% |
| 0.6-0.7 | 1 | 60.0% | 100.0% |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### deepaction · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 83 | 2.7% | 19.3% |
| 0.1-0.2 | 12 | 14.1% | 66.7% |
| 0.2-0.3 | 11 | 23.1% | 27.3% |
| 0.3-0.4 | 9 | 33.7% | 55.6% |
| 0.4-0.5 | 7 | 44.9% | 28.6% |
| 0.5-0.6 | 3 | 54.3% | 66.7% |
| 0.6-0.7 | 8 | 64.8% | 62.5% |
| 0.7-0.8 | 9 | 73.6% | 66.7% |
| 0.8-0.9 | 12 | 84.3% | 66.7% |
| 0.9-1.0 | 46 | 98.1% | 97.8% |

### defactify · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 250 | 0.7% | 5.6% |
| 0.1-0.2 | 6 | 14.4% | 33.3% |
| 0.2-0.3 | 6 | 23.6% | 66.7% |
| 0.3-0.4 | 10 | 35.1% | 70.0% |
| 0.4-0.5 | 4 | 43.9% | 100.0% |
| 0.5-0.6 | 7 | 54.9% | 85.7% |
| 0.6-0.7 | 5 | 64.1% | 80.0% |
| 0.7-0.8 | 12 | 74.9% | 91.7% |
| 0.8-0.9 | 12 | 84.2% | 83.3% |
| 0.9-1.0 | 188 | 99.1% | 100.0% |

### df26 · app as shipped

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 11 | 18.0% | 9.1% |
| 0.2-0.3 | 87 | 25.2% | 33.3% |
| 0.3-0.4 | 63 | 34.4% | 61.9% |
| 0.4-0.5 | 31 | 43.7% | 74.2% |
| 0.5-0.6 | 7 | 52.9% | 100.0% |
| 0.6-0.7 | 1 | 62.4% | 100.0% |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### df26 · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 56 | 2.7% | 23.2% |
| 0.1-0.2 | 17 | 14.1% | 23.5% |
| 0.2-0.3 | 15 | 25.8% | 26.7% |
| 0.3-0.4 | 11 | 35.5% | 45.5% |
| 0.4-0.5 | 3 | 44.6% | 66.7% |
| 0.5-0.6 | 9 | 55.3% | 33.3% |
| 0.6-0.7 | 10 | 64.8% | 30.0% |
| 0.7-0.8 | 9 | 75.5% | 55.6% |
| 0.8-0.9 | 17 | 85.9% | 58.8% |
| 0.9-1.0 | 53 | 98.4% | 96.2% |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 264 | 1.4% | 28.4% |
| 0.1-0.2 | 36 | 15.0% | 61.1% |
| 0.2-0.3 | 14 | 25.4% | 50.0% |
| 0.3-0.4 | 18 | 34.3% | 50.0% |
| 0.4-0.5 | 16 | 44.5% | 62.5% |
| 0.5-0.6 | 13 | 54.9% | 53.8% |
| 0.6-0.7 | 12 | 65.8% | 66.7% |
| 0.7-0.8 | 10 | 74.7% | 80.0% |
| 0.8-0.9 | 24 | 86.0% | 87.5% |
| 0.9-1.0 | 93 | 98.1% | 89.2% |

