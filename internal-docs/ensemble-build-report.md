## Model build parity

40 dataset images, app normalization. A logit-gap difference below ~0.05 is negligible after calibration (slopes are ~0.2).

| Check | Max \|gap diff\| | Mean | OK |
|---|---|---|---|
| commfor-224: PyTorch vs ONNX export, same input | 0.0000 | 0.0000 | yes |
| commfor-224: authors' preprocessing + PyTorch vs app preprocessing + ONNX | 0.0000 | 0.0000 | yes |
| bundled: fp32 vs fp16 weights | 0.0000 | 0.0000 | yes |
| commfor-224: fp32 vs fp16 weights | 0.0096 | 0.0028 | yes |
| ai-image-detector.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |
| commfor-224.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |

| File | fp32 MB | Shipped (fp16 weights) MB |
|---|---|---|
| ai-image-detector.onnx | 35.2 | 35.2 |
| commfor-224.onnx | 86.8 | 43.5 |
# App ensemble: bundled + Community Forensics 224

Scores come from the exact ONNX files the app bundles (fp16 weights). The ensemble score is the mean of both models' standardized logits; standardization is fit on the photo scores and shared with video: mean_d=-0.4024, std_d=7.9834, mean_c=-0.4588, std_c=6.2085.

### Photos: each model alone vs the ensemble

| Score | AUC defactify | AUC mj-dalle-sd-nbp | AI caught @5% FA defactify | AI caught @5% FA mj-dalle-sd-nbp |
|---|---|---|---|---|
| Bundled (two views) | 0.952 | 0.771 | 73.2% | 29.3% |
| Community Forensics 224 | 0.995 | 0.780 | 94.8% | 39.3% |
| **Ensemble** | 0.996 | 0.830 | 98.0% | 44.4% |

## Preprocess: `ensemble (photos)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| defactify | 1500 | 0.996 | 0.254 | 0.095 | 0.051 | 0.0% | 18.5% |
| mj-dalle-sd-nbp | 1500 | 0.830 | 0.121 | 0.158 | 0.060 | 0.0% | 3.2% |

Pooled fit: **slope = 3.3859, intercept = 0.0012**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 87.6% | 2.0% |
| 0.10 | 76.4% | 5.6% |
| 0.15 | 68.8% | 8.4% |
| 0.20 | 62.0% | 10.4% |
| 0.25 | 54.0% | 11.6% |
| 0.30 | 48.4% | 16.0% |
| 0.35 | 43.2% | 18.8% |
| 0.40 | 38.4% | 21.2% |
| 0.45 | 33.6% | 24.0% |
| 0.50 | 30.0% | 26.4% |
| 0.55 | 25.6% | 31.2% |
| 0.60 | 20.8% | 33.6% |
| 0.65 | 17.2% | 37.6% |
| 0.70 | 13.6% | 42.0% |
| 0.75 | 11.2% | 47.6% |
| 0.80 | 7.6% | 52.4% |
| 0.85 | 5.6% | 59.2% |
| 0.90 | 4.0% | 68.0% |
| 0.95 | 1.2% | 80.0% |

Recommendation: **HIGH ≥ 0.90**, **LOW < 0.15** → worst case 4.0% of real images shown HIGH (target ≤5%), 8.4% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **4.0% of real shown HIGH**, 8.4% of AI shown LOW, and at least 32.0% of AI shown HIGH.

### Video: each model alone vs the ensemble

| Score | AUC deepaction | AUC df26 | AI caught @5% FA deepaction | AI caught @5% FA df26 |
|---|---|---|---|---|
| Bundled (two views) | 0.793 | 0.651 | 41.0% | 21.0% |
| Community Forensics 224 | 0.876 | 0.854 | 52.0% | 59.0% |
| **Ensemble** | 0.892 | 0.821 | 52.0% | 54.0% |

## Preprocess: `ensemble (video)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| deepaction | 200 | 0.892 | 0.225 | 0.135 | 0.094 | 0.0% | 2.0% |
| df26 | 200 | 0.821 | 0.233 | 0.175 | 0.117 | 0.0% | 0.5% |

Pooled fit: **slope = 3.3906, intercept = -1.7718**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 100.0% | 0.0% |
| 0.10 | 97.0% | 0.0% |
| 0.15 | 90.0% | 3.0% |
| 0.20 | 83.0% | 7.0% |
| 0.25 | 73.0% | 11.0% |
| 0.30 | 67.0% | 13.0% |
| 0.35 | 57.0% | 19.0% |
| 0.40 | 47.0% | 21.0% |
| 0.45 | 44.0% | 30.0% |
| 0.50 | 32.0% | 35.0% |
| 0.55 | 30.0% | 37.0% |
| 0.60 | 26.0% | 42.0% |
| 0.65 | 23.0% | 49.0% |
| 0.70 | 19.0% | 55.0% |
| 0.75 | 7.0% | 62.0% |
| 0.80 | 4.0% | 71.0% |
| 0.85 | 2.0% | 72.0% |
| 0.90 | 0.0% | 77.0% |
| 0.95 | 0.0% | 87.0% |

Recommendation: **HIGH ≥ 0.80**, **LOW < 0.20** → worst case 4.0% of real images shown HIGH (target ≤5%), 7.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **0.0% of real shown HIGH**, 3.0% of AI shown LOW, and at least 23.0% of AI shown HIGH.

# Genned detector accuracy benchmark

Model: bundled `Dafilab/ai-image-detector` ONNX export. `squash` = what the app does today; `social` = long edge ≤1080 + JPEG q75 (Instagram re-upload). Every condition then gets the app's own normalization (long edge ≤2048, JPEG q92), as on-device. Threshold 0.5 for accuracy/FPR/FNR; bands use the app's LOW <15% / HIGH ≥90% (meaningful for calibrated rows such as 'app as shipped').

> If the model was trained on one of these datasets, its numbers there are inflated. Treat a dataset that scores far better than the others with suspicion.

## Overall

| Dataset | Model | Condition | Preprocess | n (AI/real) | AUC | Accuracy | FPR (real→AI) | FNR (AI missed) | Real shown HIGH | AI shown LOW | ECE | Scores >99% or <1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepaction | app as shipped | video | video5 | 100/100 | 0.793 | 69.5% | 17.0% | 44.0% | 0.0% | 1.0% | 0.089 | 0.0% |
| deepaction | commfor-224 (app onnx) | video | video5 | 100/100 | 0.876 | 78.0% | 18.0% | 26.0% | 5.0% | 9.0% | 0.103 | 28.5% |
| defactify | app as shipped | original | avg | 250/250 | 0.951 | 84.8% | 6.4% | 24.0% | 0.0% | 0.8% | 0.102 | 1.0% |
| defactify | app as shipped | jpeg75 | avg | 250/250 | 0.952 | 85.2% | 6.4% | 23.2% | 0.0% | 0.8% | 0.102 | 1.0% |
| defactify | app as shipped | social | avg | 250/250 | 0.952 | 85.2% | 6.4% | 23.2% | 0.0% | 0.8% | 0.102 | 1.0% |
| defactify | commfor-224 (app onnx) | original | native | 250/250 | 0.995 | 95.2% | 3.2% | 6.4% | 0.0% | 3.6% | 0.032 | 79.2% |
| defactify | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.995 | 95.6% | 2.8% | 6.0% | 0.0% | 3.6% | 0.031 | 79.4% |
| defactify | commfor-224 (app onnx) | social | native | 250/250 | 0.995 | 95.6% | 2.8% | 6.0% | 0.0% | 3.6% | 0.031 | 79.4% |
| df26 | app as shipped | video | video5 | 100/100 | 0.651 | 57.5% | 59.0% | 26.0% | 0.0% | 0.0% | 0.096 | 0.0% |
| df26 | commfor-224 (app onnx) | video | video5 | 100/100 | 0.854 | 74.5% | 32.0% | 19.0% | 6.0% | 10.0% | 0.150 | 26.5% |
| mj-dalle-sd-nbp | app as shipped | original | avg | 250/250 | 0.801 | 71.8% | 40.4% | 16.0% | 2.8% | 1.6% | 0.091 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | jpeg75 | avg | 250/250 | 0.744 | 66.4% | 40.8% | 26.4% | 2.4% | 2.8% | 0.092 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | social | avg | 250/250 | 0.770 | 68.0% | 44.8% | 19.2% | 2.8% | 1.2% | 0.091 | 0.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | original | native | 250/250 | 0.804 | 69.4% | 16.8% | 44.4% | 4.4% | 28.0% | 0.204 | 39.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.775 | 67.6% | 17.2% | 47.6% | 4.4% | 34.4% | 0.229 | 36.4% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | social | native | 250/250 | 0.759 | 66.8% | 19.2% | 47.2% | 4.4% | 32.8% | 0.232 | 34.2% |

## Per generator (bundled: squash; candidates: native; app ensemble: as shipped)

Share classified correctly at 0.5 (for `real`: share *not* flagged as AI).

### deepaction · app as shipped

| Generator | video |
|---|---|
| real | 83.0% (n=100) |
| BDAnimateDiffLightning | 94.1% (n=17) |
| CogVideoX5B | 23.5% (n=17) |
| RunwayML | 76.5% (n=17) |
| StableDiffusion | 56.2% (n=16) |
| Veo | 58.8% (n=17) |
| VideoPoet | 25.0% (n=16) |

### deepaction · commfor-224 (app onnx)

| Generator | video |
|---|---|
| real | 82.0% (n=100) |
| BDAnimateDiffLightning | 100.0% (n=17) |
| CogVideoX5B | 64.7% (n=17) |
| RunwayML | 64.7% (n=17) |
| StableDiffusion | 93.8% (n=16) |
| Veo | 58.8% (n=17) |
| VideoPoet | 62.5% (n=16) |

### defactify · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 96.8% (n=250) | 97.2% (n=250) | 97.2% (n=250) |
| dalle3 | 100.0% (n=50) | 100.0% (n=50) | 100.0% (n=50) |
| midjourney6 | 86.0% (n=50) | 86.0% (n=50) | 86.0% (n=50) |
| sd21 | 96.0% (n=50) | 96.0% (n=50) | 96.0% (n=50) |
| sd3 | 92.0% (n=50) | 94.0% (n=50) | 94.0% (n=50) |
| sdxl | 94.0% (n=50) | 94.0% (n=50) | 94.0% (n=50) |

### df26 · app as shipped

| Generator | video |
|---|---|
| real | 41.0% (n=100) |
| Grok_Imagine | 100.0% (n=10) |
| HunyuanVideo_1.5_14b_i2v | 70.0% (n=10) |
| HunyuanVideo_1.5_14b_t2v | 20.0% (n=10) |
| Kling_3.0 | 100.0% (n=10) |
| LTX_2.3_distilled_i2v | 50.0% (n=10) |
| LTX_2.3_distilled_t2v | 40.0% (n=10) |
| Veo_3.1 | 100.0% (n=10) |
| Wan_2.2_14b_i2v | 80.0% (n=10) |
| Wan_2.2_14b_t2v | 80.0% (n=10) |
| Wan_2.6 | 100.0% (n=10) |

### df26 · commfor-224 (app onnx)

| Generator | video |
|---|---|
| real | 68.0% (n=100) |
| Grok_Imagine | 100.0% (n=10) |
| HunyuanVideo_1.5_14b_i2v | 40.0% (n=10) |
| HunyuanVideo_1.5_14b_t2v | 70.0% (n=10) |
| Kling_3.0 | 100.0% (n=10) |
| LTX_2.3_distilled_i2v | 40.0% (n=10) |
| LTX_2.3_distilled_t2v | 90.0% (n=10) |
| Veo_3.1 | 100.0% (n=10) |
| Wan_2.2_14b_i2v | 70.0% (n=10) |
| Wan_2.2_14b_t2v | 100.0% (n=10) |
| Wan_2.6 | 100.0% (n=10) |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 83.2% (n=250) | 82.8% (n=250) | 80.8% (n=250) |
| ai | 55.6% (n=250) | 52.4% (n=250) | 52.8% (n=250) |

## Calibration (original, squash)

When the model says X%, how often is the image actually AI?

### deepaction · app as shipped

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 8 | 5.7% | 0.0% |
| 0.1-0.2 | 16 | 14.7% | 6.2% |
| 0.2-0.3 | 36 | 25.5% | 30.6% |
| 0.3-0.4 | 27 | 35.0% | 44.4% |
| 0.4-0.5 | 40 | 45.5% | 50.0% |
| 0.5-0.6 | 31 | 54.6% | 58.1% |
| 0.6-0.7 | 24 | 64.5% | 91.7% |
| 0.7-0.8 | 12 | 72.2% | 83.3% |
| 0.8-0.9 | 6 | 84.9% | 100.0% |
| 0.9-1.0 | 0 | n/a | n/a |

### deepaction · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 62 | 3.0% | 11.3% |
| 0.1-0.2 | 20 | 15.3% | 35.0% |
| 0.2-0.3 | 9 | 24.7% | 44.4% |
| 0.3-0.4 | 5 | 35.9% | 40.0% |
| 0.4-0.5 | 12 | 45.1% | 50.0% |
| 0.5-0.6 | 2 | 52.3% | 50.0% |
| 0.6-0.7 | 7 | 65.2% | 57.1% |
| 0.7-0.8 | 10 | 74.9% | 80.0% |
| 0.8-0.9 | 22 | 84.8% | 68.2% |
| 0.9-1.0 | 51 | 98.3% | 90.2% |

### defactify · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 235 | 0.7% | 2.1% |
| 0.1-0.2 | 9 | 13.3% | 55.6% |
| 0.2-0.3 | 9 | 25.0% | 44.4% |
| 0.3-0.4 | 3 | 36.8% | 33.3% |
| 0.4-0.5 | 2 | 49.2% | 50.0% |
| 0.5-0.6 | 3 | 52.5% | 33.3% |
| 0.6-0.7 | 3 | 66.6% | 33.3% |
| 0.7-0.8 | 4 | 75.6% | 0.0% |
| 0.8-0.9 | 6 | 82.9% | 100.0% |
| 0.9-1.0 | 226 | 99.5% | 100.0% |

### df26 · app as shipped

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 2 | 15.1% | 0.0% |
| 0.2-0.3 | 5 | 25.4% | 40.0% |
| 0.3-0.4 | 19 | 36.3% | 26.3% |
| 0.4-0.5 | 41 | 44.3% | 46.3% |
| 0.5-0.6 | 42 | 55.6% | 42.9% |
| 0.6-0.7 | 51 | 65.4% | 52.9% |
| 0.7-0.8 | 28 | 74.7% | 64.3% |
| 0.8-0.9 | 11 | 83.4% | 90.9% |
| 0.9-1.0 | 1 | 90.7% | 100.0% |

### df26 · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 49 | 3.9% | 16.3% |
| 0.1-0.2 | 15 | 14.5% | 26.7% |
| 0.2-0.3 | 11 | 26.1% | 9.1% |
| 0.3-0.4 | 8 | 33.2% | 50.0% |
| 0.4-0.5 | 4 | 44.7% | 50.0% |
| 0.5-0.6 | 9 | 54.0% | 11.1% |
| 0.6-0.7 | 7 | 66.8% | 28.6% |
| 0.7-0.8 | 14 | 75.7% | 71.4% |
| 0.8-0.9 | 16 | 84.2% | 43.8% |
| 0.9-1.0 | 67 | 98.5% | 91.0% |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 220 | 1.8% | 25.5% |
| 0.1-0.2 | 43 | 14.9% | 53.5% |
| 0.2-0.3 | 21 | 23.6% | 57.1% |
| 0.3-0.4 | 15 | 35.1% | 53.3% |
| 0.4-0.5 | 20 | 45.5% | 60.0% |
| 0.5-0.6 | 17 | 54.9% | 23.5% |
| 0.6-0.7 | 12 | 64.8% | 50.0% |
| 0.7-0.8 | 15 | 74.4% | 60.0% |
| 0.8-0.9 | 21 | 84.8% | 71.4% |
| 0.9-1.0 | 116 | 97.9% | 90.5% |

