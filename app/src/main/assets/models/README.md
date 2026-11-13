# Bundled classifier models

`AIImageClassifierProvider` combines a primary model with a Community Forensics model
(`EnsembleConfig`). Photos and video frames each have their own pair: retrained copies for
photos, the published weights for video.

| File | Model | License | Size |
|---|---|---|---|
| `ai-image-detector.onnx` | [`Dafilab/ai-image-detector`](https://huggingface.co/Dafilab/ai-image-detector) (EfficientNet-B4), **retrained for photos** in accuracy round 4 (train run 43242952132: real phone photos added, everyday-edit augmentation + universal adversarial training, open-weights generators only). Used for photos and screen captures. Input `pixel_values` `[1, 3, 380, 380]`; output: two raw logits in `[ai, human]` order. | Apache-2.0 | 35.2 MB |
| `ai-image-detector-video.onnx` | The same model as published. Used for video frames, where it beats the retrained copy on one of the two video test sets. Same input and output. | Apache-2.0 | 35.2 MB |
| `commfor-224.onnx` | Community Forensics ViT-S (CVPR 2025), **retrained for photos** in accuracy round 4 (train run 43083106402, same recipe). Used for photos and screen captures. Input `pixel_values` `[1, 3, 224, 224]`; output: one logit (AI). | MIT | 43.5 MB |
| `commfor-224-video.onnx` | [`OwensLab/commfor-model-224`](https://huggingface.co/OwensLab/commfor-model-224) as published. Used for video frames, where it beats the retrained copy. Same input and output. | MIT | 43.5 MB |

- **Weights:** all files store their weights as fp16 and cast them to fp32 at
  load, so the math is fp32. `tools/build_models.py` builds and parity-checks
  them in the `Ensemble build` workflow.
- **Delivery:** bundled into the APK at build time as app assets. They are never
  downloaded at runtime.
- **Accuracy:** see `internal-docs/MODEL.md` "Ensemble (shipped)", "Phase 3" and
  "Accuracy round 4".

To replace or re-export them, see `internal-docs/MODEL.md` in the repository
root. Do not commit a model file whose license and provenance you have not
verified.
