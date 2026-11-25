# Contributing to Genned

Thanks for helping. Genned is a small project that tries to do one thing honestly: tell
people, on their own phone, whether a photo or video is likely AI-generated. Every kind of
contribution helps: bug reports, wrong-result reports, code, models, benchmarks,
translations, design and documentation.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## The ground rules

These hold for every change and are checked in review:

1. **Images never leave the device.** The app has no internet permission, and CI fails any
   build that gains one. No uploads, analytics, crash reporting or remote models.
2. **Results are never faked.** A check that can't run reports "unavailable"; it never
   returns a made-up score. Wording stays honest: "likely", "uncertain", never "real" or
   "fake".
3. **Detection changes come with numbers.** Anything that can change a result (models,
   thresholds, calibration, preprocessing) needs before-and-after numbers from the
   benchmark workflows on held-out data, not a few hand-picked images.
4. **Licences must allow it.** Models, datasets and libraries need a licence compatible with
   Apache-2.0 and redistribution. Training data uses open-weights generators only.
5. **Privacy-relevant changes update [PRIVACY.md](PRIVACY.md)** in the same pull request.

## Ways to help

- **Report a bug** or a **wrong result** with the issue templates.
- **Pick an issue** labelled
  [`good first issue`](https://github.com/as791/genned/labels/good%20first%20issue) or
  [`help wanted`](https://github.com/as791/genned/labels/help%20wanted). Comment on it
  first so work isn't duplicated.
- **Improve detection:** propose or train better on-device models and run the benchmarks.
  See [issue #16](https://github.com/as791/genned/issues/16) and
  [`internal-docs/MODEL.md`](internal-docs/MODEL.md).
- **Translate the app:** all user-facing text is in `app/src/main/res/values/strings.xml`;
  add `values-<language>/strings.xml`.
- **Test on your phone** and report how it behaves on your device and Android version.

For anything bigger than a small fix, open an issue first to agree on the approach.

## Development setup

You need JDK 17 and the Android SDK with platform 36. Android Studio installs both.

```bash
git clone https://github.com/as791/genned.git
cd genned
./gradlew assembleDebug                    # debug APK in app/build/outputs/apk/debug/
./gradlew installDebug                     # install on a connected phone or emulator
./gradlew testDebugUnitTest :domain:test   # unit tests, including the screenshot tests
```

The screenshot tests (`app/src/test/.../ScreenshotTest.kt`) render the main screens to
`app/build/screens/`. Look at them when you change the UI.

For the Python tooling in `tools/` (benchmarks, calibration, training):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r tools/requirements.txt
python -m unittest discover -s tools/tests
```

## Where things are

| Path | What's there |
|---|---|
| `domain/` | Pure Kotlin, no Android: the evidence model and `EvidenceEngine`, which turns signals into a result. `EvidenceWeights.kt` holds the thresholds. |
| `app/.../data/detection/` | One provider per check: AI classifier, Content Credentials, generator metadata, EXIF. |
| `app/.../data/analysis/` | Photo and video analysis pipelines. |
| `app/.../ui/` | Compose screens: home, result, history, guide, settings, the bubble. |
| `app/src/main/assets/models/` | The bundled ONNX models (see the `README.md` there). |
| `tools/` | Benchmarks (`evaluate.py`, `eval_video.py`), calibration, model export and training. |
| `.github/workflows/` | CI, benchmarks (Model eval, Robustness eval, Video eval, Adversarial eval), training, the website. |
| `internal-docs/` | Architecture, model card and benchmark history, privacy design, roadmap, releasing. |

## Changing detection

1. Read [`internal-docs/MODEL.md`](internal-docs/MODEL.md): it records every model,
   benchmark and decision so far, including what didn't work.
2. Run the relevant benchmark workflow from the Actions tab on your branch (or run
   `tools/evaluate.py` locally) for both the current models and yours, on the same images.
3. Put the before-and-after table in your pull request. Mention anything that got worse.
4. Keep `tools/evaluate.py` and the Kotlin constants in sync; the code comments say which
   values mirror which.

Benchmark images stay inside the CI runners. Only numbers are uploaded.

## Pull requests

- Branch from `main` and keep each pull request to one change.
- Add or update tests for what you change. CI must be green: build, lint, unit tests, and
  the tools tests if you touched `tools/`.
- Use clear commit messages that say what changed and why.
- Update the docs your change affects: README, `internal-docs/`, `PRIVACY.md`.
- UI changes: attach before-and-after screenshots (the screenshot tests produce them).
- Don't commit secrets, signing keys or personal photos.

A maintainer reviews every pull request. Expect questions, especially about accuracy and
privacy; they're how the project keeps its promises.

## Licence

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same licence as the project.
