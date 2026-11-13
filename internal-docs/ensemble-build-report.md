## Model build parity

40 dataset images, app normalization. A logit-gap difference below ~0.05 is negligible after calibration (slopes are ~0.2).

| Check | Max \|gap diff\| | Mean | OK |
|---|---|---|---|
| commfor-224: PyTorch vs ONNX export, same input | 0.0001 | 0.0000 | yes |
| commfor-224: authors' preprocessing + PyTorch vs app preprocessing + ONNX | 0.0001 | 0.0000 | yes |
| bundled: fp32 vs fp16 weights | 0.0132 | 0.0045 | yes |
| commfor-224: fp32 vs fp16 weights | 0.0137 | 0.0037 | yes |
| ai-image-detector.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |
| commfor-224.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |

| File | fp32 MB | Shipped (fp16 weights) MB |
|---|---|---|
| ai-image-detector.onnx | 70.1 | 35.2 |
| commfor-224.onnx | 86.8 | 43.5 |
# App ensemble: bundled + Community Forensics 224

Scores come from the exact ONNX files the app bundles (fp16 weights). The ensemble score is the mean of both models' standardized logits; standardization is fit on the photo scores and shared with video: mean_d=-0.0905, std_d=3.4163, mean_c=-3.9895, std_c=4.3740.

### Photos: each model alone vs the ensemble

| Score | AUC defactify | AUC mj-dalle-sd-nbp | AI caught @5% FA defactify | AI caught @5% FA mj-dalle-sd-nbp |
|---|---|---|---|---|
| Bundled (two views) | 0.957 | 0.855 | 78.1% | 49.5% |
| Community Forensics 224 | 0.959 | 0.646 | 83.3% | 21.5% |
| **Ensemble** | 0.989 | 0.820 | 96.0% | 44.7% |

## Preprocess: `ensemble (photos)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| defactify | 1500 | 0.989 | 0.256 | 0.110 | 0.064 | 0.0% | 20.5% |
| mj-dalle-sd-nbp | 1500 | 0.820 | 0.115 | 0.130 | 0.055 | 0.0% | 4.7% |

Pooled fit: **slope = 3.5233, intercept = 0.3752**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 90.8% | 1.2% |
| 0.10 | 77.2% | 5.6% |
| 0.15 | 61.2% | 10.0% |
| 0.20 | 52.8% | 14.8% |
| 0.25 | 46.4% | 18.0% |
| 0.30 | 40.0% | 21.2% |
| 0.35 | 33.2% | 25.2% |
| 0.40 | 29.6% | 27.2% |
| 0.45 | 24.4% | 32.4% |
| 0.50 | 19.6% | 36.4% |
| 0.55 | 17.6% | 40.4% |
| 0.60 | 14.4% | 44.4% |
| 0.65 | 12.0% | 47.6% |
| 0.70 | 9.6% | 51.2% |
| 0.75 | 8.4% | 55.2% |
| 0.80 | 6.4% | 59.6% |
| 0.85 | 4.0% | 64.8% |
| 0.90 | 3.6% | 72.4% |
| 0.95 | 2.0% | 79.6% |

Recommendation: **HIGH ≥ 0.85**, **LOW < 0.15** → worst case 4.0% of real images shown HIGH (target ≤5%), 10.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.20), worst case: **3.6% of real shown HIGH**, 14.8% of AI shown LOW, and at least 27.6% of AI shown HIGH.

### Video: each model alone vs the ensemble

| Score | AUC deepaction | AUC df26 | AI caught @5% FA deepaction | AI caught @5% FA df26 |
|---|---|---|---|---|
| Bundled (two views) | 0.731 | 0.764 | 26.0% | 20.0% |
| Community Forensics 224 | 0.970 | 0.743 | 70.0% | 33.0% |
| **Ensemble** | 0.939 | 0.785 | 74.0% | 35.0% |

## Preprocess: `ensemble (video)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| deepaction | 200 | 0.939 | 0.232 | 0.112 | 0.071 | 0.0% | 4.5% |
| df26 | 200 | 0.785 | 0.132 | 0.129 | 0.083 | 0.0% | 1.5% |

Pooled fit: **slope = 3.2227, intercept = -1.2816**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 97.0% | 0.0% |
| 0.10 | 89.0% | 1.0% |
| 0.15 | 76.0% | 5.0% |
| 0.20 | 64.0% | 8.0% |
| 0.25 | 55.0% | 11.0% |
| 0.30 | 48.0% | 18.0% |
| 0.35 | 39.0% | 24.0% |
| 0.40 | 33.0% | 28.0% |
| 0.45 | 28.0% | 29.0% |
| 0.50 | 27.0% | 29.0% |
| 0.55 | 22.0% | 34.0% |
| 0.60 | 17.0% | 41.0% |
| 0.65 | 13.0% | 45.0% |
| 0.70 | 12.0% | 49.0% |
| 0.75 | 11.0% | 55.0% |
| 0.80 | 10.0% | 59.0% |
| 0.85 | 6.0% | 64.0% |
| 0.90 | 3.0% | 71.0% |
| 0.95 | 0.0% | 80.0% |

Recommendation: **HIGH ≥ 0.90**, **LOW < 0.20** → worst case 3.0% of real images shown HIGH (target ≤5%), 8.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **3.0% of real shown HIGH**, 5.0% of AI shown LOW, and at least 29.0% of AI shown HIGH.

# Genned detector accuracy benchmark

Model: bundled `Dafilab/ai-image-detector` ONNX export. `squash` = what the app does today; `social` = long edge ≤1080 + JPEG q75 (Instagram re-upload). Every condition then gets the app's own normalization (long edge ≤2048, JPEG q92), as on-device. Threshold 0.5 for accuracy/FPR/FNR; bands use the app's LOW <20% / HIGH ≥90% (meaningful for calibrated rows such as 'app as shipped').

> If the model was trained on one of these datasets, its numbers there are inflated. Treat a dataset that scores far better than the others with suspicion.

## Overall

| Dataset | Model | Condition | Preprocess | n (AI/real) | AUC | Accuracy | FPR (real→AI) | FNR (AI missed) | Real shown HIGH | AI shown LOW | ECE | Scores >99% or <1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepaction | app as shipped | video | video5 | 100/100 | 0.731 | 52.0% | 0.0% | 96.0% | 0.0% | 1.0% | 0.249 | 0.0% |
| deepaction | commfor-224 (app onnx) | video | video5 | 100/100 | 0.970 | 78.5% | 0.0% | 43.0% | 0.0% | 26.0% | 0.197 | 38.5% |
| defactify | app as shipped | original | avg | 250/250 | 0.957 | 86.6% | 8.0% | 18.8% | 0.0% | 0.0% | 0.208 | 0.0% |
| defactify | app as shipped | jpeg75 | avg | 250/250 | 0.957 | 86.8% | 8.0% | 18.4% | 0.0% | 0.0% | 0.209 | 0.0% |
| defactify | app as shipped | social | avg | 250/250 | 0.957 | 86.8% | 8.0% | 18.4% | 0.0% | 0.0% | 0.209 | 0.0% |
| defactify | commfor-224 (app onnx) | original | native | 250/250 | 0.960 | 78.4% | 0.8% | 42.4% | 0.0% | 33.6% | 0.209 | 54.8% |
| defactify | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.959 | 78.0% | 1.2% | 42.8% | 0.0% | 33.6% | 0.208 | 54.0% |
| defactify | commfor-224 (app onnx) | social | native | 250/250 | 0.959 | 78.0% | 1.2% | 42.8% | 0.0% | 33.6% | 0.208 | 54.0% |
| df26 | app as shipped | video | video5 | 100/100 | 0.764 | 54.0% | 0.0% | 92.0% | 0.0% | 0.0% | 0.192 | 0.0% |
| df26 | commfor-224 (app onnx) | video | video5 | 100/100 | 0.743 | 61.5% | 3.0% | 74.0% | 0.0% | 65.0% | 0.353 | 49.5% |
| mj-dalle-sd-nbp | app as shipped | original | avg | 250/250 | 0.874 | 80.2% | 24.4% | 15.2% | 0.0% | 0.0% | 0.175 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | jpeg75 | avg | 250/250 | 0.837 | 76.6% | 24.0% | 22.8% | 0.0% | 0.0% | 0.143 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | social | avg | 250/250 | 0.855 | 76.4% | 29.2% | 18.0% | 0.0% | 0.0% | 0.140 | 0.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | original | native | 250/250 | 0.700 | 56.8% | 1.2% | 85.2% | 0.8% | 78.8% | 0.408 | 64.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.616 | 54.2% | 1.2% | 90.4% | 0.4% | 84.8% | 0.441 | 67.4% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | social | native | 250/250 | 0.621 | 54.8% | 1.2% | 89.2% | 0.8% | 83.6% | 0.438 | 69.8% |

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
| real | 100.0% (n=100) |
| BDAnimateDiffLightning | 100.0% (n=17) |
| CogVideoX5B | 35.3% (n=17) |
| RunwayML | 23.5% (n=17) |
| StableDiffusion | 81.2% (n=16) |
| Veo | 47.1% (n=17) |
| VideoPoet | 56.2% (n=16) |

### defactify · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 99.2% (n=250) | 98.8% (n=250) | 98.8% (n=250) |
| dalle3 | 76.0% (n=50) | 76.0% (n=50) | 76.0% (n=50) |
| midjourney6 | 36.0% (n=50) | 36.0% (n=50) | 36.0% (n=50) |
| sd21 | 78.0% (n=50) | 78.0% (n=50) | 78.0% (n=50) |
| sd3 | 42.0% (n=50) | 42.0% (n=50) | 42.0% (n=50) |
| sdxl | 56.0% (n=50) | 54.0% (n=50) | 54.0% (n=50) |

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
| real | 97.0% (n=100) |
| Grok_Imagine | 40.0% (n=10) |
| HunyuanVideo_1.5_14b_i2v | 10.0% (n=10) |
| HunyuanVideo_1.5_14b_t2v | 30.0% (n=10) |
| Kling_3.0 | 30.0% (n=10) |
| LTX_2.3_distilled_i2v | 10.0% (n=10) |
| LTX_2.3_distilled_t2v | 0.0% (n=10) |
| Veo_3.1 | 30.0% (n=10) |
| Wan_2.2_14b_i2v | 10.0% (n=10) |
| Wan_2.2_14b_t2v | 30.0% (n=10) |
| Wan_2.6 | 70.0% (n=10) |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 98.8% (n=250) | 98.8% (n=250) | 98.8% (n=250) |
| ai | 14.8% (n=250) | 9.6% (n=250) | 10.8% (n=250) |

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
| 0.0-0.1 | 109 | 2.1% | 15.6% |
| 0.1-0.2 | 11 | 12.6% | 90.9% |
| 0.2-0.3 | 10 | 24.2% | 50.0% |
| 0.3-0.4 | 7 | 35.2% | 71.4% |
| 0.4-0.5 | 6 | 45.1% | 100.0% |
| 0.5-0.6 | 8 | 55.1% | 100.0% |
| 0.6-0.7 | 1 | 69.6% | 100.0% |
| 0.7-0.8 | 10 | 74.7% | 100.0% |
| 0.8-0.9 | 5 | 84.9% | 100.0% |
| 0.9-1.0 | 33 | 98.7% | 100.0% |

### defactify · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 306 | 1.2% | 20.6% |
| 0.1-0.2 | 24 | 14.8% | 87.5% |
| 0.2-0.3 | 10 | 26.5% | 90.0% |
| 0.3-0.4 | 4 | 33.1% | 100.0% |
| 0.4-0.5 | 10 | 44.7% | 90.0% |
| 0.5-0.6 | 10 | 54.7% | 90.0% |
| 0.6-0.7 | 9 | 65.4% | 88.9% |
| 0.7-0.8 | 15 | 75.6% | 100.0% |
| 0.8-0.9 | 20 | 86.0% | 100.0% |
| 0.9-1.0 | 92 | 97.9% | 100.0% |

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
| 0.0-0.1 | 153 | 1.5% | 41.2% |
| 0.1-0.2 | 7 | 13.5% | 42.9% |
| 0.2-0.3 | 5 | 25.7% | 60.0% |
| 0.3-0.4 | 3 | 33.9% | 66.7% |
| 0.4-0.5 | 3 | 42.7% | 100.0% |
| 0.5-0.6 | 4 | 54.1% | 75.0% |
| 0.6-0.7 | 4 | 62.1% | 75.0% |
| 0.7-0.8 | 2 | 76.7% | 50.0% |
| 0.8-0.9 | 6 | 85.8% | 100.0% |
| 0.9-1.0 | 13 | 95.8% | 100.0% |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 415 | 1.0% | 43.6% |
| 0.1-0.2 | 24 | 15.0% | 66.7% |
| 0.2-0.3 | 11 | 23.8% | 72.7% |
| 0.3-0.4 | 5 | 35.2% | 100.0% |
| 0.4-0.5 | 5 | 45.6% | 60.0% |
| 0.5-0.6 | 5 | 54.1% | 100.0% |
| 0.6-0.7 | 1 | 65.5% | 100.0% |
| 0.7-0.8 | 4 | 74.1% | 100.0% |
| 0.8-0.9 | 7 | 85.2% | 85.7% |
| 0.9-1.0 | 23 | 97.7% | 91.3% |

