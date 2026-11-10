# Architecture

## Data flow

```
content:// URI (share sheet / photo picker)
        |
        v
ImageLoader.normalize()
  - copies original bytes verbatim (EXIF/PNG chunks/C2PA stay intact)
  - decodes a bounded, orientation-corrected preview/classifier JPEG
        |
        v
AnalysisInput (domain model: two file paths + basic dimensions, no Android types)
        |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
  provenance group      metadata group         visual group
  (C2PAProvider)   (Exif/KnownGeneratorMetadata)  (AIImageClassifier, Watermark)
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                       EvidenceEngine.aggregate()
                              |
                              v
                   AnalysisResult (score, classification, signals, limitations)
                              |
                    +---------+---------+
                    v                   v
              Room (history)      Result screen
```

`AnalyzeImageUseCase` (`app/data/analysis`) runs the three provider groups in that
order, reporting each group's start as a real `AnalysisStage` — the Analyzing
screen's stage text reflects actual pipeline progress, never a simulated timer.

## Modules

- **`domain`** — pure Kotlin, no Android dependency, compiles and tests as a plain
  JVM module. Holds `DetectionProvider`, `DetectionSignal`, `AnalysisResult`,
  `Classification`, and `EvidenceEngine`. Deliberately isolated so the
  highest-risk logic (evidence aggregation) can be tested without an Android SDK
  and recalibrated later without touching UI or provider code.
- **`app`** — everything Android: Compose UI (`ui/`), provider implementations
  (`data/detection/*`), Room persistence (`data/storage`), image normalization
  (`data/image`), and share-card rendering (`data/sharing`).

A single Gradle module was deliberately *not* used for everything: keeping
`domain` Android-free is what makes `EvidenceEngineTest` runnable without an
Android SDK/emulator — a real constraint during this project's own development
sandbox, and a genuine ongoing benefit for fast local iteration.

## Detection providers

Every detector implements the same interface:

```kotlin
interface DetectionProvider {
    val signalType: SignalType
    suspend fun analyze(image: AnalysisInput): DetectionSignal
}
```

Implemented in V1:
- `ExifMetadataProvider` — presence of camera-capture EXIF fields (make/model/
  exposure). Weak, one-directional signal (see EvidenceWeights below).
- `KnownGeneratorMetadataProvider` — scans EXIF and PNG tEXt/iTXt chunks for known
  generator signatures (`GeneratorSignatures.kt`). A match is real evidence; no
  match is not evidence of anything (metadata is trivially stripped).
- `AIImageClassifierProvider` — ONNX Runtime Mobile inference. See `internal-docs/MODEL.md`.

Implemented as honest stubs (`SignalAvailability.UNAVAILABLE`, not fabricated):
- `C2PAProvider` — see "Content Credentials (C2PA)" below.
- `WatermarkProvider` — no open on-device detector exists for generative
  watermarks (e.g. SynthID); see its class doc.

Adding a new detector (a better classifier, a real C2PA check, a watermark model)
means writing one new `DetectionProvider` and adding it to one of the three
provider-group lists in `AppContainer` — nothing else changes.

## Video (Reels/Shorts)

Handled by a parallel pipeline, `AnalyzeVideoUseCase` (`app/data/analysis`), not
by extending `AnalyzeImageUseCase`:

```
video Uri
   |
   v
VideoFrameSampler (MediaMetadataRetriever, N evenly-spaced frames -> JPEGs)
   |
   v
AIImageClassifierProvider.analyze() per frame (same provider as photos)
   |
   v
VideoSignalAggregator.aggregateFrameSignals() -> one AI_CLASSIFIER DetectionSignal
   |
   v
EvidenceEngine.aggregate() (same engine, same thresholds, plus explicit
UNAVAILABLE stubs for metadata/provenance and a video-specific limitation string)
```

This reuses the *same* on-device image classifier per frame rather than a
video-specific model — there isn't one bundled, and none is implied. It is
explicitly frame-sampled still-image classification, not motion, temporal, or
audio analysis; every video result's limitations say this in plain language
(`AnalyzeVideoUseCase.videoLimitation`). `VideoSignalAggregator` is pure Kotlin
(only depends on domain models), so the frame-averaging logic is unit-tested
without needing a real video file or Robolectric.

Content Credentials are checked on the video file itself (see "Content Credentials
(C2PA)"). EXIF and generator-signature inspection are not implemented for video —
rather than silently omitting those signal cards, `AnalyzeVideoUseCase` includes
explicit `UNAVAILABLE` `DetectionSignal`s for them so the Result screen states the
gap honestly instead of looking incomplete.

### Share-sheet routing for image vs. video

`ShareIntentParser.extractSharedMedia` resolves both `ACTION_SEND` and
`ACTION_SEND_MULTIPLE` (some share panels — notably Samsung's Gallery "Share via"
sheet — dispatch `SEND_MULTIPLE` even for a single selected item, so a
SEND-only filter simply never appears as a target from those sources) to a
`SharedMedia(uri, kind)`, where `kind` is derived from the intent's MIME type
(`image/*` vs `video/*`). `MainActivity` passes `kind` through as a nav argument
so `AnalyzingViewModel` can route to `AnalyzeImageUseCase` or
`AnalyzeVideoUseCase` accordingly.

## Evidence engine

`EvidenceEngine.aggregate()` has two paths:

1. **Verified provenance available** (a `CONTENT_CREDENTIALS` signal is
   `AVAILABLE`): its score *is* the result, not blended with probabilistic
   signals — matching "cryptographically verified provenance should be presented
   separately from probabilistic ML detection." Reached only by Content Credentials
   that verify against the bundled trust lists (see "Content Credentials (C2PA)").
2. **Probabilistic blend** (the normal V1 path): a confidence-weighted average
   over every `AVAILABLE` signal, using fixed weights in `EvidenceWeights.kt`.

### Why corroborating signals scale their weight by their own score

The classifier is bidirectional by design (trained to discriminate both ways), so
its weight is flat regardless of the score it returns. Metadata/watermark signals
are one-directional: *finding* a generator signature is real evidence, but *not*
finding one is merely an absence of evidence. An earlier version of this engine
gave these signals a flat weight regardless of outcome, which meant a confirmed
"no generator metadata found" diluted the classifier's contribution and quietly
pulled every result toward "human" — caught by
`EvidenceEngineTest."absent generator metadata never pulls likelihood toward AI or
toward human"`. The fix: a corroborating signal's weight is scaled by its own
score, so "not found" (score 0) contributes zero weight rather than diluting
anything.

### Recalibrating the evidence weights

`EvidenceWeights.kt` is a documented starting point, not a validated calibration.
Once `tools/evaluate.py` has been run against a real labeled dataset with the
actual bundled classifier, use those precision/recall numbers to inform:
- `CLASSIFIER_WEIGHT` relative to the metadata weights,
- `HIGH_THRESHOLD` / `LOW_THRESHOLD` (currently 0.70 / 0.30, chosen to leave a
  deliberately wide UNCERTAIN band rather than force a confident-sounding verdict
  from an unvalidated classifier).

This is isolated in one object specifically so it can change without touching
providers, the use case, or the UI.

## Content Credentials (C2PA)

`C2PAProvider` reads and verifies a file's embedded Content Credentials with
`contentauth/c2pa-android` (MIT/Apache-2.0, the C2PA Rust SDK over JNI), fetched from
JitPack (`com.github.contentauth:c2pa-android`, limited to that group in
`settings.gradle.kts`) because the project's own Maven channel needs a GitHub token.

- **Android 9+ only.** The library's minSdk is 28; the app keeps 26 through
  `tools:overrideLibrary` and an SDK check before any call. Release builds leave out its
  x86/x86_64 native libraries (~25 MB each); there the check reports "not available".
- **What it reads:** the byte-exact original (`originalFilePath`), or the video file
  straight from its URI. The file type is sniffed from its first bytes.
- **Verification is fully local.** `C2paTrustSettings` turns trust checking on, turns
  remote-manifest fetching off, and supplies trust anchors bundled in
  `app/src/main/assets/c2pa/`:
  - `c2pa-trust-list.pem`: the official C2PA Trust List
    (github.com/c2pa-org/conformance-public, `trust-list/C2PA-TRUST-LIST.pem`, CC-BY-4.0).
  - `interim-trust-list.pem`: the Content Authenticity Initiative's interim trust list
    (contentcredentials.org/trust/anchors.pem), which content signed before the C2PA
    list existed chains to (Truepic, Adobe, Microsoft, camera makers).
  - `trust-config.cfg`: the accepted certificate purposes (contentcredentials.org/trust/store.cfg).

  These are byte-identical copies (SHA-256 in `assets/c2pa/SHA256SUMS`). Refresh them
  before a release by downloading the same URLs.
- **Mapping** (`C2paManifestInterpreter`, pure Kotlin, tested against real c2patool
  reports in `app/src/test/resources/c2pa/`):

  | Manifest | Signal |
  |---|---|
  | Trusted signer, generative AI declared anywhere (IPTC `trainedAlgorithmicMedia`, `compositeWithTrainedAlgorithmicMedia`) | AVAILABLE, 0.98 |
  | Trusted signer, camera capture declared, no AI anywhere | AVAILABLE, 0.05 |
  | Trusted signer, neither declared | UNAVAILABLE |
  | Signer not on the trust lists | UNAVAILABLE (an AI claim is mentioned, never counted) |
  | Fails verification (changed after signing, bad signature or certificate) | UNAVAILABLE |
  | None, online-only, unsupported type, old Android, timeout (15 s) | UNAVAILABLE |

  An AVAILABLE signal is authoritative in `EvidenceEngine` (verified-provenance branch),
  so only a manifest that verifies against the bundled trust lists can produce it.
- **On-device test:** `.github/workflows/c2pa-device-test.yml` signs fixtures with a
  throwaway test CA and runs `C2paDeviceTest` on an emulator, so the native reader, the
  stream adapter and the error mapping are checked against the real library.

## Screen overlay (experimental)

Off by default, opt-in only, toggled from Settings -> Experimental. Exists to reach
two cases nothing else in this app can: content Instagram/WhatsApp only let you
forward *within* the app itself (never through Android's share sheet, so
`ShareIntentParser` never sees it), and checking a Reel/photo while scrolling
without manually sharing each one out.

```
Settings toggle
   |  (chained permission requests, one system dialog at a time)
   v
1. Settings.canDrawOverlays()          -- "draw over other apps"
2. AppOpsManager usage-access check    -- "usage access"
3. MediaProjectionManager consent      -- one-time per service start
   |
   v
OverlayCaptureService (foreground service, type=mediaProjection)
   |
   +-- ForegroundAppWatcher: polls UsageStatsManager every 1.5s for the
   |   current foreground package; bubble is shown only over
   |   TARGET_PACKAGES (Instagram, WhatsApp) and hidden everywhere else
   |
   +-- BubbleView: a draggable floating window (plain View/Canvas, not
       Compose -- see its class doc for why). Tap it:
          |
          v
       captureFrame() -- grabs the current frame from the MediaProjection's
       ImageReader (the *rendered pixels* of whatever is on screen),
       encodes to JPEG in cacheDir
          |
          v
       AnalyzeImageUseCase.run() -- the exact same pipeline as a normal
       share-in check (classifier + metadata + provenance signals)
          |
          v
       bubble shows the resulting AI-likelihood percentage; tapping again
       opens the full Result screen (MainActivity via
       Routes.EXTRA_OPEN_ANALYSIS_ID)
```

### Why MediaProjection, not Accessibility Service

The obvious alternative — Accessibility Service — was deliberately rejected, even
though it's the more common approach for "overlay on top of another app" tools.
Accessibility Service would let (and would be flagged by Play Store review as
being able to) read Instagram/WhatsApp's actual view hierarchy and text content,
which is exactly the kind of automated in-app content scraping this project's own
constraints rule out. MediaProjection instead captures *rendered pixels only* —
the same class of access screen recorders and screenshot tools use, sanctioned by
Google for this purpose, requiring an explicit, per-session, revocable user
consent dialog every time (Android will not let this be silently
re-granted). `ForegroundAppWatcher` uses `UsageStatsManager`, not Accessibility
Service either — it only ever learns a foreground package *name*, never any
content within it.

### Why tap-to-capture, not continuous analysis

The bubble analyzes only on an explicit tap. Continuously analyzing every frame
while scrolling was considered and rejected for V1: it would run the classifier
several times a second indefinitely, with real battery cost, and would mean the
app is analyzing content the user never asked it to look at — a materially
different privacy posture than "capture what's on screen right now, once, because
I tapped a button." Tap-to-capture keeps the same consent-per-action model as
the rest of the app (you always initiate an analysis).

### The capture surface is created once per grant, attached only during a tap

The `VirtualDisplay`/`ImageReader` is created once, right after the capture grant,
**with no surface attached**, and kept for the whole overlay session. A tap attaches
the reader (`setSurface(reader.surface)`), hides the bubble (which also makes Android
compose a fresh frame), reads one frame, and detaches it again (`setSurface(null)`)
in a `finally` (`CaptureSurfaceGate`, unit-tested). It is released only when the
overlay stops.

Why only during a tap: while a surface is attached, Android copies every frame of the
full-resolution screen into it, so an earlier version that attached it for the whole
Instagram/WhatsApp visit spent battery on every scrolled frame even if you never
tapped. Now scrolling costs no capture work at all. `ForegroundAppWatcher` also stops
polling while the screen is off (screen on/off broadcasts) and catches up on the next
poll.

Why not create one per tap or per visit: Android 14+ allows only **one**
`createVirtualDisplay()` call per `MediaProjection` grant. A second call
revokes the grant (`MediaProjection.Callback.onStop()`), which killed the
overlay on-device — first on the 2nd tap (when displays were per-tap), then on
re-entering Instagram (when they were per-visit), both confirmed from debug
logs on a real device. A `Service` can't silently re-request consent (that
needs an `Activity` to show the system dialog), so the one display has to be
reused. Detaching its surface is like turning that virtual screen off, so
nothing is mirrored — no GPU/battery cost — between taps.

### System-revoked capture stops the whole overlay, not just that tap

`MediaProjection.Callback.onStop()` (registered in `startProjection`) can fire on
its own, not only when the user taps the system's "stop sharing" control — seen
on-device as the bubble silently disappearing mid-tap, with a capture that never
completes. Since there's no way to tell from inside that callback whether the OS
revoked capture for a legitimate reason (screen off, an app-lifecycle/idle
timeout, or another device-specific policy) versus an actual bug, `onStop()`
tears down the whole service via `stopSelf()` rather than trying to guess and
silently retry — but it now also logs distinctly from a user-requested
`ACTION_STOP` and shows a toast, because leaving the user to notice a vanished
bubble with no explanation is worse than a one-line "capture permission ended,
turn it back on" message. Re-enabling from Settings gets a fresh consent grant.

### What this does not do

- Does not read Instagram/WhatsApp's UI, database, or network traffic — only a
  screen-pixel capture, on tap, of whatever is currently rendered. The screen is
  not mirrored at all while you scroll; only for the moment of a tap.
- Does not run unless explicitly turned on in Settings, and stops the moment the
  toggle is turned off or the persistent notification's "Stop" action is tapped —
  a foreground-service notification is mandatory and always visible while active
  (Android requirement, not a choice this app makes).
- Does not persist the captured screen frame beyond the single analysis it was
  captured for (deleted after; see `internal-docs/PRIVACY.md`).
- Is not published to any Play Store listing claim as a "silent" or "background"
  feature — see `internal-docs/PLAY_STORE_CHECKLIST.md` for the disclosure this requires
  before any store submission.

## Why no DI framework

The object graph (`AppContainer`) is small, static for the process lifetime, and
fits on one screen. Hilt/Koin would add a build-time dependency, annotation
processing, and a layer of generated code for a graph a single engineer can read
end-to-end faster than tracing DI-generated code. Revisit only if the graph grows
enough that manual wiring becomes the harder-to-read option.

## Monetization-ready seams

`domain/model/Entitlement.kt` defines `Entitlement { FREE, PRO }` and an
`EntitlementProvider` interface with a single `AlwaysFreeEntitlementProvider`
implementation wired in V1. No feature actually checks entitlement yet — this
exists so a future Pro tier (unlimited analysis, advanced models, batch analysis)
can gate behind one new `EntitlementProvider` implementation instead of scattered
`if` checks. See `internal-docs/ROADMAP.md`.
