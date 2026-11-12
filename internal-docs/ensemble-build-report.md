## Model build parity

40 dataset images, app normalization. A logit-gap difference below ~0.05 is negligible after calibration (slopes are ~0.2).

| Check | Max \|gap diff\| | Mean | OK |
|---|---|---|---|
| commfor-224: PyTorch vs ONNX export, same input | 0.0000 | 0.0000 | yes |
| commfor-224: authors' preprocessing + PyTorch vs app preprocessing + ONNX | 0.0000 | 0.0000 | yes |
| bundled: fp32 vs fp16 weights | 0.0000 | 0.0000 | yes |
| commfor-224: fp32 vs fp16 weights | 0.0100 | 0.0033 | yes |
| ai-image-detector.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |
| commfor-224.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |

| File | fp32 MB | Shipped (fp16 weights) MB |
|---|---|---|
| ai-image-detector.onnx | 35.2 | 35.2 |
| commfor-224.onnx | 86.8 | 43.5 |
# App ensemble: bundled + Community Forensics 224

Scores come from the exact ONNX files the app bundles (fp16 weights). The ensemble score is the mean of both models' standardized logits; standardization is fit on the photo scores and shared with video: mean_d=-0.4024, std_d=7.9834, mean_c=-1.4826, std_c=6.0167.

### Photos: each model alone vs the ensemble

| Score | AUC defactify | AUC mj-dalle-sd-nbp | AI caught @5% FA defactify | AI caught @5% FA mj-dalle-sd-nbp |
|---|---|---|---|---|
| Bundled (two views) | 0.952 | 0.771 | 73.2% | 29.3% |
| Community Forensics 224 | 0.986 | 0.794 | 94.4% | 36.3% |
| **Ensemble** | 0.994 | 0.836 | 97.6% | 46.5% |

## Preprocess: `ensemble (photos)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| defactify | 1500 | 0.994 | 0.248 | 0.102 | 0.060 | 0.0% | 17.4% |
| mj-dalle-sd-nbp | 1500 | 0.836 | 0.123 | 0.168 | 0.070 | 0.0% | 4.2% |

Pooled fit: **slope = 3.3570, intercept = 0.0458**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 85.2% | 2.4% |
| 0.10 | 76.0% | 5.6% |
| 0.15 | 67.2% | 8.0% |
| 0.20 | 59.6% | 10.4% |
| 0.25 | 54.8% | 11.6% |
| 0.30 | 50.8% | 14.4% |
| 0.35 | 43.6% | 17.2% |
| 0.40 | 39.6% | 21.2% |
| 0.45 | 34.4% | 22.4% |
| 0.50 | 28.8% | 24.8% |
| 0.55 | 24.0% | 28.8% |
| 0.60 | 19.6% | 31.2% |
| 0.65 | 16.8% | 34.4% |
| 0.70 | 13.6% | 37.6% |
| 0.75 | 10.4% | 45.6% |
| 0.80 | 8.4% | 52.0% |
| 0.85 | 5.6% | 56.4% |
| 0.90 | 3.6% | 64.8% |
| 0.95 | 2.0% | 77.6% |

Recommendation: **HIGH ≥ 0.90**, **LOW < 0.15** → worst case 3.6% of real images shown HIGH (target ≤5%), 8.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.20), worst case: **3.6% of real shown HIGH**, 10.4% of AI shown LOW, and at least 35.2% of AI shown HIGH.

### Video: each model alone vs the ensemble

| Score | AUC deepaction | AUC df26 | AI caught @5% FA deepaction | AI caught @5% FA df26 |
|---|---|---|---|---|
| Bundled (two views) | 0.793 | 0.651 | 41.0% | 21.0% |
| Community Forensics 224 | 0.868 | 0.826 | 53.0% | 57.0% |
| **Ensemble** | 0.893 | 0.801 | 53.0% | 47.0% |

## Preprocess: `ensemble (video)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| deepaction | 200 | 0.893 | 0.205 | 0.154 | 0.092 | 0.0% | 1.0% |
| df26 | 200 | 0.801 | 0.198 | 0.197 | 0.119 | 0.0% | 0.0% |

Pooled fit: **slope = 3.2139, intercept = -1.7497**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 99.0% | 0.0% |
| 0.10 | 97.0% | 0.0% |
| 0.15 | 93.0% | 1.0% |
| 0.20 | 84.0% | 6.0% |
| 0.25 | 76.0% | 10.0% |
| 0.30 | 70.0% | 13.0% |
| 0.35 | 62.0% | 20.0% |
| 0.40 | 56.0% | 26.0% |
| 0.45 | 46.0% | 33.0% |
| 0.50 | 35.0% | 35.0% |
| 0.55 | 32.0% | 38.0% |
| 0.60 | 27.0% | 40.0% |
| 0.65 | 18.0% | 48.0% |
| 0.70 | 14.0% | 55.0% |
| 0.75 | 10.0% | 63.0% |
| 0.80 | 4.0% | 68.0% |
| 0.85 | 0.0% | 72.0% |
| 0.90 | 0.0% | 79.0% |
| 0.95 | 0.0% | 89.0% |

Recommendation: **HIGH ≥ 0.80**, **LOW < 0.25** → worst case 4.0% of real images shown HIGH (target ≤5%), 10.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **0.0% of real shown HIGH**, 1.0% of AI shown LOW, and at least 21.0% of AI shown HIGH.

# Genned detector accuracy benchmark

Model: bundled `Dafilab/ai-image-detector` ONNX export. `squash` = what the app does today; `social` = long edge ≤1080 + JPEG q75 (Instagram re-upload). Every condition then gets the app's own normalization (long edge ≤2048, JPEG q92), as on-device. Threshold 0.5 for accuracy/FPR/FNR; bands use the app's LOW <20% / HIGH ≥90% (meaningful for calibrated rows such as 'app as shipped').

> If the model was trained on one of these datasets, its numbers there are inflated. Treat a dataset that scores far better than the others with suspicion.

## Overall

| Dataset | Model | Condition | Preprocess | n (AI/real) | AUC | Accuracy | FPR (real→AI) | FNR (AI missed) | Real shown HIGH | AI shown LOW | ECE | Scores >99% or <1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepaction | app as shipped | video | video5 | 100/100 | 0.793 | 69.5% | 17.0% | 44.0% | 0.0% | 1.0% | 0.089 | 0.0% |
| deepaction | commfor-224 (app onnx) | video | video5 | 100/100 | 0.868 | 77.0% | 12.0% | 34.0% | 1.0% | 20.0% | 0.135 | 31.5% |
| defactify | app as shipped | original | avg | 250/250 | 0.951 | 84.8% | 6.4% | 24.0% | 0.0% | 4.0% | 0.102 | 1.0% |
| defactify | app as shipped | jpeg75 | avg | 250/250 | 0.952 | 85.2% | 6.4% | 23.2% | 0.0% | 4.0% | 0.102 | 1.0% |
| defactify | app as shipped | social | avg | 250/250 | 0.952 | 85.2% | 6.4% | 23.2% | 0.0% | 4.0% | 0.102 | 1.0% |
| defactify | commfor-224 (app onnx) | original | native | 250/250 | 0.986 | 92.8% | 2.0% | 12.4% | 0.0% | 6.4% | 0.057 | 70.0% |
| defactify | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.986 | 92.6% | 1.6% | 13.2% | 0.0% | 6.4% | 0.058 | 70.2% |
| defactify | commfor-224 (app onnx) | social | native | 250/250 | 0.986 | 92.6% | 1.6% | 13.2% | 0.0% | 6.4% | 0.058 | 70.2% |
| df26 | app as shipped | video | video5 | 100/100 | 0.651 | 57.5% | 59.0% | 26.0% | 0.0% | 0.0% | 0.096 | 0.0% |
| df26 | commfor-224 (app onnx) | video | video5 | 100/100 | 0.826 | 73.0% | 26.0% | 28.0% | 2.0% | 15.0% | 0.140 | 25.5% |
| mj-dalle-sd-nbp | app as shipped | original | avg | 250/250 | 0.801 | 71.8% | 40.4% | 16.0% | 2.8% | 2.8% | 0.091 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | jpeg75 | avg | 250/250 | 0.744 | 66.4% | 40.8% | 26.4% | 2.4% | 5.2% | 0.092 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | social | avg | 250/250 | 0.770 | 68.0% | 44.8% | 19.2% | 2.8% | 2.0% | 0.091 | 0.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | original | native | 250/250 | 0.812 | 70.4% | 10.0% | 49.2% | 4.0% | 38.8% | 0.213 | 44.8% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.794 | 68.2% | 10.0% | 53.6% | 4.4% | 42.0% | 0.246 | 45.2% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | social | native | 250/250 | 0.775 | 66.6% | 12.0% | 54.8% | 4.0% | 41.6% | 0.244 | 43.6% |

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

