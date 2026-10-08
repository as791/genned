## Model build parity

40 dataset images, app normalization. A logit-gap difference below ~0.05 is negligible after calibration (slopes are ~0.2).

| Check | Max \|gap diff\| | Mean | OK |
|---|---|---|---|
| commfor-224: PyTorch vs ONNX export, same input | 0.0000 | 0.0000 | yes |
| commfor-224: authors' preprocessing + PyTorch vs app preprocessing + ONNX | 0.0000 | 0.0000 | yes |
| bundled: fp32 vs fp16 weights | 0.0000 | 0.0000 | yes |
| commfor-224: fp32 vs fp16 weights | 0.0086 | 0.0028 | yes |
| ai-image-detector.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |
| commfor-224.onnx: loadable by the app's ONNX Runtime 1.19 (IR 8 ≤ 10, opset 17 ≤ 21) | – | – | yes |

| File | fp32 MB | Shipped (fp16 weights) MB |
|---|---|---|
| ai-image-detector.onnx | 35.2 | 35.2 |
| commfor-224.onnx | 86.8 | 43.5 |
# App ensemble: bundled + Community Forensics 224

Scores come from the exact ONNX files the app bundles (fp16 weights). The ensemble score is the mean of both models' standardized logits; standardization is fit on the photo scores and shared with video: mean_d=-0.4024, std_d=7.9834, mean_c=-1.2687, std_c=6.0045.

### Photos: each model alone vs the ensemble

| Score | AUC defactify | AUC mj-dalle-sd-nbp | AI caught @5% FA defactify | AI caught @5% FA mj-dalle-sd-nbp |
|---|---|---|---|---|
| Bundled (two views) | 0.952 | 0.771 | 73.2% | 29.3% |
| Community Forensics 224 | 0.982 | 0.786 | 92.1% | 37.3% |
| **Ensemble** | 0.994 | 0.831 | 96.8% | 45.3% |

## Preprocess: `ensemble (photos)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| defactify | 1500 | 0.994 | 0.236 | 0.103 | 0.064 | 0.0% | 15.9% |
| mj-dalle-sd-nbp | 1500 | 0.831 | 0.120 | 0.186 | 0.074 | 0.0% | 3.3% |

Pooled fit: **slope = 3.2645, intercept = 0.0257**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 89.6% | 1.6% |
| 0.10 | 76.4% | 4.4% |
| 0.15 | 70.8% | 7.6% |
| 0.20 | 63.2% | 9.2% |
| 0.25 | 56.4% | 10.8% |
| 0.30 | 52.0% | 13.2% |
| 0.35 | 44.4% | 16.8% |
| 0.40 | 39.2% | 17.6% |
| 0.45 | 35.2% | 21.6% |
| 0.50 | 31.2% | 25.6% |
| 0.55 | 27.6% | 28.8% |
| 0.60 | 23.6% | 31.2% |
| 0.65 | 17.2% | 35.6% |
| 0.70 | 14.4% | 41.6% |
| 0.75 | 12.0% | 46.0% |
| 0.80 | 8.4% | 51.6% |
| 0.85 | 5.6% | 58.4% |
| 0.90 | 4.0% | 66.0% |
| 0.95 | 1.6% | 79.6% |

Recommendation: **HIGH ≥ 0.90**, **LOW < 0.20** → worst case 4.0% of real images shown HIGH (target ≤5%), 9.2% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **4.0% of real shown HIGH**, 7.6% of AI shown LOW, and at least 34.0% of AI shown HIGH.

### Video: each model alone vs the ensemble

| Score | AUC deepaction | AUC df26 | AI caught @5% FA deepaction | AI caught @5% FA df26 |
|---|---|---|---|---|
| Bundled (two views) | 0.793 | 0.651 | 41.0% | 21.0% |
| Community Forensics 224 | 0.859 | 0.849 | 48.0% | 57.0% |
| **Ensemble** | 0.881 | 0.818 | 51.0% | 55.0% |

## Preprocess: `ensemble (video)`

| Dataset | n | AUC | ECE raw | ECE fit-on-other-dataset | ECE pooled fit | >99%/<1% raw | >99%/<1% calibrated |
|---|---|---|---|---|---|---|---|
| deepaction | 200 | 0.881 | 0.185 | 0.141 | 0.084 | 0.0% | 1.0% |
| df26 | 200 | 0.818 | 0.203 | 0.182 | 0.117 | 0.0% | 0.5% |

Pooled fit: **slope = 3.1938, intercept = -1.8533**  (P(ai) = sigmoid(slope · (ai − human) + intercept))

Worst case over every dataset × condition, calibrated scores:

| Threshold | Real images ≥ threshold (would show HIGH) | AI images < threshold (would show LOW) |
|---|---|---|
| 0.05 | 100.0% | 0.0% |
| 0.10 | 97.0% | 1.0% |
| 0.15 | 92.0% | 3.0% |
| 0.20 | 86.0% | 6.0% |
| 0.25 | 77.0% | 11.0% |
| 0.30 | 70.0% | 12.0% |
| 0.35 | 58.0% | 22.0% |
| 0.40 | 52.0% | 27.0% |
| 0.45 | 43.0% | 30.0% |
| 0.50 | 34.0% | 37.0% |
| 0.55 | 30.0% | 40.0% |
| 0.60 | 25.0% | 45.0% |
| 0.65 | 23.0% | 50.0% |
| 0.70 | 19.0% | 56.0% |
| 0.75 | 5.0% | 62.0% |
| 0.80 | 2.0% | 71.0% |
| 0.85 | 2.0% | 72.0% |
| 0.90 | 0.0% | 75.0% |
| 0.95 | 0.0% | 88.0% |

Recommendation: **HIGH ≥ 0.75**, **LOW < 0.20** → worst case 5.0% of real images shown HIGH (target ≤5%), 6.0% of AI images shown LOW (target ≤10%).

At the app's bands (HIGH ≥ 0.90, LOW < 0.15), worst case: **0.0% of real shown HIGH**, 3.0% of AI shown LOW, and at least 25.0% of AI shown HIGH.

# Genned detector accuracy benchmark

Model: bundled `Dafilab/ai-image-detector` ONNX export. `squash` = what the app does today; `social` = long edge ≤1080 + JPEG q75 (Instagram re-upload). Every condition then gets the app's own normalization (long edge ≤2048, JPEG q92), as on-device. Threshold 0.5 for accuracy/FPR/FNR; bands use the app's LOW <15% / HIGH ≥90% (meaningful for calibrated rows such as 'app as shipped').

> If the model was trained on one of these datasets, its numbers there are inflated. Treat a dataset that scores far better than the others with suspicion.

## Overall

| Dataset | Model | Condition | Preprocess | n (AI/real) | AUC | Accuracy | FPR (real→AI) | FNR (AI missed) | Real shown HIGH | AI shown LOW | ECE | Scores >99% or <1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepaction | app as shipped | video | video5 | 100/100 | 0.793 | 69.5% | 17.0% | 44.0% | 0.0% | 1.0% | 0.089 | 0.0% |
| deepaction | commfor-224 (app onnx) | video | video5 | 100/100 | 0.859 | 76.5% | 16.0% | 31.0% | 5.0% | 14.0% | 0.129 | 27.5% |
| defactify | app as shipped | original | avg | 250/250 | 0.951 | 84.8% | 6.4% | 24.0% | 0.0% | 0.8% | 0.102 | 1.0% |
| defactify | app as shipped | jpeg75 | avg | 250/250 | 0.952 | 85.2% | 6.4% | 23.2% | 0.0% | 0.8% | 0.102 | 1.0% |
| defactify | app as shipped | social | avg | 250/250 | 0.952 | 85.2% | 6.4% | 23.2% | 0.0% | 0.8% | 0.102 | 1.0% |
| defactify | commfor-224 (app onnx) | original | native | 250/250 | 0.982 | 90.8% | 2.0% | 16.4% | 0.0% | 7.6% | 0.071 | 75.4% |
| defactify | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.982 | 90.8% | 2.0% | 16.4% | 0.0% | 8.0% | 0.069 | 75.8% |
| defactify | commfor-224 (app onnx) | social | native | 250/250 | 0.982 | 90.8% | 2.0% | 16.4% | 0.0% | 8.0% | 0.069 | 75.8% |
| df26 | app as shipped | video | video5 | 100/100 | 0.651 | 57.5% | 59.0% | 26.0% | 0.0% | 0.0% | 0.096 | 0.0% |
| df26 | commfor-224 (app onnx) | video | video5 | 100/100 | 0.849 | 75.5% | 30.0% | 19.0% | 6.0% | 10.0% | 0.127 | 25.5% |
| mj-dalle-sd-nbp | app as shipped | original | avg | 250/250 | 0.801 | 71.8% | 40.4% | 16.0% | 2.8% | 1.6% | 0.091 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | jpeg75 | avg | 250/250 | 0.744 | 66.4% | 40.8% | 26.4% | 2.4% | 2.8% | 0.092 | 0.0% |
| mj-dalle-sd-nbp | app as shipped | social | avg | 250/250 | 0.770 | 68.0% | 44.8% | 19.2% | 2.8% | 1.2% | 0.091 | 0.0% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | original | native | 250/250 | 0.810 | 71.4% | 12.4% | 44.8% | 3.6% | 33.2% | 0.200 | 41.4% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | jpeg75 | native | 250/250 | 0.780 | 67.0% | 13.2% | 52.8% | 3.6% | 37.6% | 0.235 | 38.8% |
| mj-dalle-sd-nbp | commfor-224 (app onnx) | social | native | 250/250 | 0.766 | 67.0% | 13.6% | 52.4% | 3.6% | 38.4% | 0.238 | 37.6% |

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
| real | 84.0% (n=100) |
| BDAnimateDiffLightning | 100.0% (n=17) |
| CogVideoX5B | 58.8% (n=17) |
| RunwayML | 70.6% (n=17) |
| StableDiffusion | 93.8% (n=16) |
| Veo | 47.1% (n=17) |
| VideoPoet | 43.8% (n=16) |

### defactify · commfor-224 (app onnx)

| Generator | original | jpeg75 | social |
|---|---|---|---|
| real | 98.0% (n=250) | 98.0% (n=250) | 98.0% (n=250) |
| dalle3 | 72.0% (n=50) | 72.0% (n=50) | 72.0% (n=50) |
| midjourney6 | 54.0% (n=50) | 54.0% (n=50) | 54.0% (n=50) |
| sd21 | 98.0% (n=50) | 98.0% (n=50) | 98.0% (n=50) |
| sd3 | 96.0% (n=50) | 96.0% (n=50) | 96.0% (n=50) |
| sdxl | 98.0% (n=50) | 98.0% (n=50) | 98.0% (n=50) |

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
| real | 70.0% (n=100) |
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
| real | 87.6% (n=250) | 86.8% (n=250) | 86.4% (n=250) |
| ai | 55.2% (n=250) | 47.2% (n=250) | 47.6% (n=250) |

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
| 0.0-0.1 | 68 | 3.0% | 13.2% |
| 0.1-0.2 | 18 | 14.6% | 44.4% |
| 0.2-0.3 | 10 | 26.6% | 50.0% |
| 0.3-0.4 | 11 | 35.9% | 45.5% |
| 0.4-0.5 | 8 | 43.7% | 50.0% |
| 0.5-0.6 | 8 | 53.7% | 75.0% |
| 0.6-0.7 | 2 | 64.1% | 50.0% |
| 0.7-0.8 | 13 | 77.3% | 69.2% |
| 0.8-0.9 | 9 | 84.9% | 55.6% |
| 0.9-1.0 | 53 | 98.1% | 90.6% |

### defactify · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 250 | 0.5% | 6.0% |
| 0.1-0.2 | 10 | 14.1% | 70.0% |
| 0.2-0.3 | 10 | 25.2% | 70.0% |
| 0.3-0.4 | 8 | 35.7% | 62.5% |
| 0.4-0.5 | 8 | 45.3% | 87.5% |
| 0.5-0.6 | 5 | 55.7% | 60.0% |
| 0.6-0.7 | 7 | 65.4% | 100.0% |
| 0.7-0.8 | 9 | 76.4% | 66.7% |
| 0.8-0.9 | 11 | 84.5% | 100.0% |
| 0.9-1.0 | 182 | 99.4% | 100.0% |

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
| 0.0-0.1 | 47 | 3.9% | 12.8% |
| 0.1-0.2 | 15 | 14.3% | 33.3% |
| 0.2-0.3 | 13 | 24.4% | 30.8% |
| 0.3-0.4 | 9 | 33.5% | 22.2% |
| 0.4-0.5 | 5 | 43.9% | 40.0% |
| 0.5-0.6 | 7 | 54.8% | 0.0% |
| 0.6-0.7 | 11 | 63.8% | 63.6% |
| 0.7-0.8 | 13 | 75.0% | 69.2% |
| 0.8-0.9 | 16 | 85.3% | 43.8% |
| 0.9-1.0 | 64 | 98.5% | 90.6% |

### mj-dalle-sd-nbp · commfor-224 (app onnx)

| Score bin | n | Mean score | Actually AI |
|---|---|---|---|
| 0.0-0.1 | 249 | 2.0% | 28.5% |
| 0.1-0.2 | 33 | 14.5% | 51.5% |
| 0.2-0.3 | 18 | 25.9% | 50.0% |
| 0.3-0.4 | 17 | 34.8% | 47.1% |
| 0.4-0.5 | 14 | 45.5% | 50.0% |
| 0.5-0.6 | 12 | 56.4% | 41.7% |
| 0.6-0.7 | 14 | 65.7% | 71.4% |
| 0.7-0.8 | 19 | 74.4% | 78.9% |
| 0.8-0.9 | 22 | 86.1% | 68.2% |
| 0.9-1.0 | 102 | 98.1% | 91.2% |

