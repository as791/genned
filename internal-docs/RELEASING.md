# Releasing Genned

## Release builds

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
3. The next CI run produces a signed release APK and a signed AAB (the AAB is for Play).

The keystore is written to the runner's temp directory for the build and deleted
afterwards. Each CI build gets `versionCode` = 1000 + run number, so successive builds
install and upload as updates.

**Keep the upload key backed up.** Releases are installed directly from GitHub, so this key
is also the key every installed copy trusts. Without it, no update can ever install over
an existing copy. If the app goes to Play later, enrol in Play App Signing with **this same
key** (Play Console's "use a key from Java keystore" option, not a Google-generated key).
Then GitHub and Play installs can update each other.

## Publishing a release

1. Add user-facing notes in `release-notes/<version>.md` on `main`. The release page shows
   them under "What's new".
2. In GitHub, open Actions → Android CI → Run workflow and enter a version such as `0.6.0`,
   or push a tag:
   ```bash
   git tag v0.6.0 && git push origin v0.6.0
   ```

Either way, the same CI workflow builds, tests and signs the release, then creates a GitHub
release `v0.6.0` with the universal `genned-0.6.0.apk`, one APK per CPU type
(`genned-0.6.0-arm64-v8a.apk`, `-armeabi-v7a`, `-x86_64`, `-x86`) and `SHA256SUMS.txt`.
The version must look like `MAJOR.MINOR.PATCH`; it sets `versionName`. `versionCode` keeps
counting with the workflow's runs (each APK uses it × 10 plus a per-CPU offset), so any
release APK installs over any earlier one. The step fails if the upload-key secrets are
missing, so an unsigned APK is never published. The AAB and `mapping.txt` stay in that
run's `genned-release` artifact.

After a green Android CI run on `main`, the Website workflow rebuilds
[as791.github.io/genned](https://as791.github.io/genned/) with the new version, download
links and app screens.

## Replacing a model

The bundled models live in `app/src/main/assets/models/` (a pair for photos and a pair for
video; see the `README.md` there). To replace one, put a verified `.onnx` file at the same
path and refit its constants as described in `MODEL.md` "Replacing the model file".
`tools/convert_model.py` documents exporting the original model from Hugging Face, and
`tools/evaluate.py` and the benchmark workflows measure the result. A model is **never**
downloaded at runtime; it is only ever bundled at build time.
