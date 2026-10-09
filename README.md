# Genned

A private, on-device Android app that estimates whether an image — or a shared
video like a Reel or Short — is likely AI-generated. Share it in from
Instagram, X, Reddit, WhatsApp, a browser, or your Gallery, and get an
evidence-based likelihood estimate in seconds, along with the reasoning behind
it — never a claim of certainty.

> **This is an estimate, not proof.** AI-content detection can produce false
> positives and false negatives. See "Known accuracy limitations" below.

## Download

Genned isn't on Google Play yet. Install it from
[GitHub Releases](https://github.com/as791/genned/releases/latest) (Android 8.0+):
download `genned-<version>.apk` on your phone, open it, and allow installs from that app
when Android asks. Each release lists the APK's SHA-256 and the signing certificate's
fingerprint. The app has no internet permission, so it can't tell you about updates. Watch
the repository's releases instead, and install a newer APK over the old one.

Privacy: everything runs on your phone and nothing is collected. See the
[privacy policy](PRIVACY.md).

## What it does

1. You share an image into Genned (or pick one directly from the app).
2. It's analyzed entirely on your device: a visual AI-likelihood classifier,
   metadata inspection (EXIF, PNG generator signatures), and provenance checks.
3. You see an AI-likelihood percentage and a classification —
   **HIGH** / **UNCERTAIN** / **LOW** (deliberately never "REAL"/"FAKE"/"definitely"
   anything, unless cryptographically verified provenance actually supports it),
   plus a "Why?" breakdown of every signal that contributed.
4. The result — and only the result, never your original image — can be saved
   to local history or shared as a card via the normal Android share sheet.

No accounts, no backend, no cloud sync, no ads, no subscriptions.

## User flow

```
Home ──choose image──▶ Analyzing ──▶ Result ──▶ History
  ▲                                     │           │
  └───────── "Check Another" ───────────┘           │
  ▲                                                  │
  └──────────────── tap a saved row ─────────────────┘

Any app's Share sheet ──▶ Genned ──▶ Analyzing ──▶ Result
```

Screens: **Home** (choose image / share hint / recent checks), **Analyzing**
(preview + real pipeline-stage progress, never a fake timer), **Result** (score,
classification, per-signal "Why?" cards, disclaimer, share/check-another),
**History** (thumbnail rows, per-item delete, clear-all), **Settings** (privacy
statement, bundled-model status, version).

## How detection works

Genned never relies on a single classifier. Multiple independent evidence
sources each report a score — or explicitly "unavailable," never a fabricated
value — and those are combined transparently into one likelihood estimate:

- **Visual AI classifier** — on-device ONNX Runtime inference with the bundled
  `Dafilab/ai-image-detector` model (EfficientNet-B4, exported to ONNX).
  A shared screenshot (file named "Screenshot…" or the size of the phone's screen, and
  no camera EXIF) is classified on the picture inside it, not the surrounding app UI.
- **Generator metadata** — scans EXIF and PNG text chunks for known
  generative-tool signatures (Stable Diffusion/ComfyUI "parameters" chunks,
  Midjourney/DALL·E/Firefly software strings, etc.). A match is real evidence;
  no match is never treated as evidence the image is authentic (this metadata
  is trivially stripped by re-saving or sharing).
- **EXIF camera metadata** — presence of camera capture fields (make/model/
  exposure) is weak evidence *for* a real photo; its absence is treated as
  near-zero evidence, since screenshots and social re-uploads strip EXIF
  constantly too.
- **Content Credentials (C2PA)** — interface implemented, honestly reports
  "unavailable" in this build. If ever enabled, a cryptographically valid
  manifest *overrides* the probabilistic blend rather than being averaged
  into it.
- **Known watermark detection** — interface implemented, honestly reports
  "unavailable": no open, on-device detector exists for generative watermarks
  like SynthID as of this writing.

**Video (Reels/Shorts):** shared video is handled by sampling a handful of
evenly-spaced still frames and running the *same* image classifier on each
one, then averaging the scores — this is frame-sampled still-image
classification, not motion/temporal or audio analysis, and every video result
says so explicitly.

**Screen overlay (experimental, off by default):** a floating bubble, enabled
from Settings → Experimental, that lets you check whatever is currently on
screen inside Instagram or WhatsApp — including content those apps only let
you forward internally, which no share-sheet integration can reach. It
captures on-screen frames via Android's `MediaProjection` API (the same
sanctioned mechanism screen recorders use) only while the bubble is visible,
and analyzes only when you tap it — never automatically, and never reading
the other app's actual content.

## Build

Requires JDK 17+ and an Android SDK (via Android Studio, or `sdkmanager`) with
the `compileSdk 35` platform + build tools installed.

```bash
./gradlew assembleDebug
```

Install to a connected device/emulator:

```bash
./gradlew installDebug
```

Run tests:

```bash
./gradlew testDebugUnitTest :domain:test
```

### Release builds

CI builds the **release** variant on every push (`lintRelease`, `assembleRelease`,
`bundleRelease`), so R8 code shrinking and resource shrinking are always exercised. The
outputs and R8's `mapping.txt` are uploaded as the `genned-release` artifact. Release
builds strip all debug-level logging and hide the debug-log tools.

Release builds are signed only with a **private upload key**. The key is never committed;
the committed `app/debug.keystore` signs debug builds only. Without the key, the release
build is produced unsigned. To sign it:

1. Create an upload key once, on your own machine, and back it up somewhere safe:
   ```bash
   keytool -genkeypair -v -keystore genned-upload.jks -alias genned-upload \
     -keyalg RSA -keysize 4096 -validity 10000
   base64 -w0 genned-upload.jks > genned-upload.jks.b64   # macOS: base64 -i genned-upload.jks
   ```
2. In the GitHub repository, open Settings → Secrets and variables → Actions, and add:
   - `UPLOAD_KEYSTORE_BASE64`: the contents of `genned-upload.jks.b64`;
   - `UPLOAD_STORE_PASSWORD`;
   - `UPLOAD_KEY_ALIAS` (here `genned-upload`);
   - `UPLOAD_KEY_PASSWORD`.
3. The next CI run produces a signed release APK and a signed AAB (the AAB is for Play,
   later).

The keystore is written to the runner's temp directory for the build and deleted
afterwards. Each CI build gets `versionCode` = 1000 + run number, so successive builds
install and upload as updates.

**Keep the upload key backed up.** Releases are installed directly from GitHub, so this key
is also the key every installed copy trusts. Without it, no update can ever install over
an existing copy. If the app goes to Play later, enrol in Play App Signing with **this same
key** (Play Console's "use a key from Java keystore" option, not a Google-generated key).
Then GitHub and Play installs can update each other.

### Publishing a release

In GitHub, open Actions → Android CI → Run workflow and enter a version such as `0.2.0`,
or push a tag:

```bash
git tag v0.2.0 && git push origin v0.2.0
```

Either way, the same CI workflow builds, tests and signs the release, then creates a GitHub
release `v0.2.0` with `genned-0.2.0.apk` and `SHA256SUMS.txt`. The version must look like
`MAJOR.MINOR.PATCH`; it sets `versionName`. `versionCode` keeps
counting with the workflow's runs, so each release installs over the previous one. The
step fails if the upload-key secrets are missing, so an unsigned APK is never published.
The AAB and `mapping.txt` stay in that run's `genned-release` artifact.

## How to replace/update the ML model

The bundled model lives at `app/src/main/assets/models/ai-image-detector.onnx`.
To replace it, drop a verified `.onnx` file at that path — nothing else needs to
change; the classifier and Settings screen pick it up automatically.
`tools/convert_model.py` documents exporting the current model
(`Dafilab/ai-image-detector`, Apache-2.0) from Hugging Face, and
`tools/evaluate.py` measures accuracy/precision/recall/F1/confusion matrix
against a labeled dataset. The model is **never** downloaded at runtime —
only ever bundled at build time.

## Licenses

- App source code: [Apache-2.0](LICENSE).
- Kotlin, Jetpack Compose, AndroidX libraries (Room, Navigation, ExifInterface,
  Activity/Lifecycle): Apache-2.0.
- ONNX Runtime Mobile (`com.microsoft.onnxruntime:onnxruntime-android`): MIT.
- Bundled classifier models: `Dafilab/ai-image-detector` (Apache-2.0) and
  Community Forensics ViT-S 224 (`OwensLab/commfor-model-224`, MIT; Park & Owens, CVPR 2025),
  as published and as a copy fine-tuned for photos by this project (MIT). The fine-tuning
  used images from OpenFake (`ComplexDataLab/OpenFake`, CC-BY-SA-4.0, open-weights
  generators only) and the Defactify dataset.
- `contentauth/c2pa-android` (referenced, not bundled): dual MIT/Apache-2.0.

## Known accuracy limitations

- **The visual classifier ships with the app.** It is an ensemble of two
  on-device models, `Dafilab/ai-image-detector` and Community Forensics ViT-S
  224, in `app/src/main/assets/models/` (about 122 MB). Photos use a copy of
  Community Forensics fine-tuned on newer generators and everyday edits; video
  keeps the published weights, which are better on video. Combined, the models
  beat either one alone. On the harder photo benchmark they catch 45% of AI
  images at a 5% false-alarm rate (the single model caught 30%); on the easier
  one, 97% (it caught 73%). If a model file is ever missing from a build, the
  app falls back to the remaining model, or reports the classifier as
  unavailable, rather than faking a score.
- **Measured accuracy is moderate.** On two public real-vs-AI image datasets
  (500 images each), the bundled classifier reaches an AUC of 0.91 and 0.80
  (1.0 = perfect, 0.5 = coin flip). On the harder set, about a third of real
  images were scored as likely AI, and DALL·E 3 images were caught only about
  half the time. Its raw scores were far too confident, so the app now
  calibrates them (see below). Evaluating stronger models is the current focus:
  see
  [issue #14](https://github.com/as791/genned/issues/14) and
  `internal-docs/MODEL.md` "Measured accuracy". Anyone can re-run the benchmark
  from the repo's Actions tab ("Model eval").
- **Video is less reliable than photos.** Reels and Shorts are analyzed by
  running the image classifier on 5 sampled frames. There is no motion or audio
  analysis. On two recent real-vs-AI video benchmarks the image model scored
  many *real* videos as AI-like (26% of real talking-head clips came out HIGH).
  Video now has its own calibration, and the ensemble's second model is much
  stronger on video. On two benchmarks it catches 82% and 30% of AI videos at
  a 5% false-alarm rate (the single model caught 41% and 21%), and at most 2%
  of real videos show HIGH. Modern talking-head fakes remain hard.
- **Not robust to deliberate attacks.** An attacker with the model files can
  add invisible noise that flips the result. #15 measured this and tried the
  phone-feasible defenses: a consistency check and three rounds of adversarial
  fine-tuning. None held up against an attacker who adapts, so none ships.
  The app's job is to help you judge ordinary content, including recompressed
  and reshared images, not to resist a determined adversary.
- Like every AI-image detector, the classifier's training data has a cutoff
  and will be weaker against newer generators; compression, screenshotting,
  and intentional adversarial editing can all shift results in either
  direction.
- **Scores are calibrated, and bands are set to avoid false alarms.** The raw
  classifier's scores are rescaled to match how often it's actually right, and
  "HIGH" needs 90% or more, and "LOW" needs under 15%. In the benchmark,
  under 3% of real images were labeled HIGH, and at most 10% of AI images were
  labeled LOW. The trade-off is that many images honestly come out UNCERTAIN.

## Contributing

Genned is early, and help is very welcome, especially from people working on AI
provenance, computer vision or Android. Good places to start:

- [#14 Detector accuracy](https://github.com/as791/genned/issues/14): calibrate
  the scores, and benchmark and propose better on-device detectors.
- [#15 Adversarial robustness](https://github.com/as791/genned/issues/15):
  an open problem. The benchmarks and the fine-tuning pipeline
  (`notebooks/adversarial_finetune.ipynb`) are in place. Making these detectors
  robust likely needs an adversarially pretrained backbone and real GPU time.
- [#1 Public release readiness](https://github.com/as791/genned/issues/1): the
  checklist for a store release.

Any change to detection should come with numbers from the `Model eval` workflow
(`tools/evaluate.py`), not just anecdotes. Images never leave the device, and
changes must keep it that way.

## Documentation

Deeper technical notes for contributors — architecture, the model card, the
privacy design, the roadmap, and a pre-release checklist — live in
[`internal-docs/`](internal-docs/), kept separate from this README so it stays
focused on what the app does and how to build it.
