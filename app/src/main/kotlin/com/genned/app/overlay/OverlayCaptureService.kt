package com.genned.app.overlay

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.core.content.IntentCompat
import com.genned.app.GennedApplication
import com.genned.app.MainActivity
import com.genned.app.R
import com.genned.app.data.storage.MediaKind
import com.genned.app.text.displayPercent
import com.genned.app.ui.navigation.Routes
import com.genned.domain.model.AnalysisInput
import com.genned.domain.model.Classification
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.roundToInt
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Experimental, opt-in-only foreground service backing the floating "Screen
 * Overlay" bubble (Settings -> Experimental). See internal-docs/ARCHITECTURE.md "Screen
 * overlay (experimental)" for the full design and why it captures the screen via
 * [MediaProjection] rather than reading other apps' content via Accessibility
 * Service.
 *
 * Lifecycle: started only from SettingsScreen, only after the user has explicitly
 * granted the overlay-draw permission, usage-access permission, and a
 * MediaProjection consent for this session. Stopped the moment the user turns the
 * experiment off (or taps Stop on the persistent notification this foreground
 * service is required to show), or if the MediaProjection is revoked by the
 * system. Nothing here runs at boot or starts itself.
 */
class OverlayCaptureService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var analysisJob: Job? = null
    private var watchJob: Job? = null

    private lateinit var windowManager: WindowManager
    private var bubbleView: BubbleView? = null
    private var dismissTarget: DismissTargetView? = null
    private var layoutParams: WindowManager.LayoutParams? = null
    private var lastAnalysisId: Long? = null

    private var mediaProjection: MediaProjection? = null
    private var projectionCallback: MediaProjection.Callback? = null
    private var captureReader: ImageReader? = null
    private var captureDisplay: VirtualDisplay? = null

    /** Connects the mirror only during a tap; see CaptureSurfaceGate and ARCHITECTURE.md. */
    private val captureGate = CaptureSurfaceGate(
        attach = {
            Log.d(TAG, "capture surface attach at ${System.currentTimeMillis()}")
            captureDisplay?.setSurface(captureReader?.surface)
        },
        detach = {
            Log.d(TAG, "capture surface detach at ${System.currentTimeMillis()}")
            captureDisplay?.setSurface(null)
        },
    )

    /** Screen on/off, so foreground-app polling stops while the phone is in your pocket. */
    private val screenOn = MutableStateFlow(true)
    private var screenReceiver: BroadcastReceiver? = null

    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDensity = 0
    private var screenTopInset = 0
    private var screenBottomInset = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            Log.i(TAG, "onStartCommand: ACTION_STOP received, stopping")
            stopSelf()
            return START_NOT_STICKY
        }

        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0
        val data = intent?.let { IntentCompat.getParcelableExtra(it, EXTRA_RESULT_DATA, Intent::class.java) }
        if (data == null || resultCode == 0) {
            Log.w(TAG, "onStartCommand: missing MediaProjection result data, stopping without starting")
            stopSelf()
            return START_NOT_STICKY
        }

        if (mediaProjection != null) {
            Log.i(TAG, "onStartCommand: already running, tearing down the previous session before restarting")
            tearDownSession()
        }

        Log.i(TAG, "onStartCommand: starting overlay (foreground service + projection + bubble)")
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            buildNotification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
        )
        if (!startProjection(resultCode, data)) {
            stopSelf()
            return START_NOT_STICKY
        }
        addBubble()
        watchScreenState()
        watchForegroundApp()
        _running.value = true

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy: stopping overlay (this is the moment battery cost should end)")
        tearDownSession()
        serviceScope.cancel()
        _running.value = false
        super.onDestroy()
    }

    private fun tearDownSession() {
        watchJob?.cancel()
        watchJob = null
        screenReceiver?.let { runCatching { unregisterReceiver(it) } }
        screenReceiver = null
        analysisJob?.cancel()
        analysisJob = null
        removeBubble()
        closeCaptureSurface()
        stopProjection()
    }

    // --- Foreground-app awareness: show the bubble specifically over the apps
    // this experiment targets, hide it elsewhere. See ForegroundAppWatcher. ---

    private fun watchForegroundApp() {
        watchJob = serviceScope.launch {
            ForegroundAppWatcher(this@OverlayCaptureService).watch(screenOn).collect { packageName ->
                val shouldShow = packageName != null && packageName in ForegroundAppWatcher.TARGET_PACKAGES
                Log.d(TAG, "foregroundPackage=$packageName shouldShowBubble=$shouldShow")
                // Only the bubble: the screen mirror stays disconnected until a tap
                // (CaptureSurfaceGate), so scrolling these apps costs no capture work.
                bubbleView?.visibility = if (shouldShow) android.view.View.VISIBLE else android.view.View.GONE
            }
        }
    }

    private fun watchScreenState() {
        screenOn.value = (getSystemService(Context.POWER_SERVICE) as PowerManager).isInteractive
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                screenOn.value = intent.action == Intent.ACTION_SCREEN_ON
            }
        }
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
        }
        ContextCompat.registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        screenReceiver = receiver
    }

    // --- MediaProjection setup ---
    //
    // The MediaProjection consent (below) is obtained once and held for the whole
    // service lifetime, and so is the VirtualDisplay/ImageReader it feeds: Android
    // 14+ allows only ONE createVirtualDisplay() call per MediaProjection grant - a
    // second call revokes the grant (MediaProjection.Callback.onStop()), seen
    // on-device as the overlay dying on the 2nd tap (when displays were per-tap)
    // and then on re-entering Instagram (when they were per-visit). A Service can't
    // silently re-request consent (that needs an Activity to show the system
    // dialog), so the display is created once, right after the grant, with no
    // surface attached. A tap attaches the ImageReader for the moment of the capture
    // and detaches it again (CaptureSurfaceGate): with no surface, nothing is
    // mirrored, so there is no copying cost while you scroll, with or without the
    // bubble showing. It's released only when the overlay stops.

    /** Starts screen capture; false if the system refused the grant (nothing to clean up). */
    private fun startProjection(resultCode: Int, data: Intent): Boolean {
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getRealMetrics(metrics)
        screenWidth = metrics.widthPixels
        screenHeight = metrics.heightPixels
        screenDensity = metrics.densityDpi
        // A Service isn't a UI context; if the platform refuses window metrics here, crop without insets.
        val (topInset, bottomInset) = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            runCatching {
                val insets = windowManager.currentWindowMetrics.windowInsets
                    .getInsetsIgnoringVisibility(WindowInsets.Type.systemBars())
                insets.top to insets.bottom
            }.getOrNull() ?: (0 to 0)
        } else {
            0 to 0
        }
        screenTopInset = topInset
        screenBottomInset = bottomInset

        val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val projection = projectionManager.getMediaProjection(resultCode, data)
        if (projection == null) {
            Log.w(TAG, "getMediaProjection returned null: the capture grant was not accepted")
            return false
        }
        mediaProjection = projection

        // Required since Android 14 (API 34): createVirtualDisplay() throws
        // IllegalStateException if no callback is registered first. onStop fires
        // when the user revokes capture from the system's "stop sharing" control,
        // not just when we call stop() ourselves - without handling it here, every
        // capture attempt after a system-initiated revoke would crash instead of
        // failing gracefully.
        //
        // It also fires unprompted on some devices/OS versions well before any user
        // action - seen on-device as the whole overlay silently vanishing mid-tap
        // (openCaptureSurface immediately followed by onDestroy, no error shown -
        // see internal-docs/ARCHITECTURE.md "Screen overlay (experimental)"). Since we can't
        // distinguish "user tapped the system stop-sharing control" from "the OS
        // revoked this MediaProjection on its own" from inside this callback, at
        // least stop being silent about it: log it distinctly from a
        // user-requested ACTION_STOP, and tell the user directly, since otherwise
        // the bubble just disappears with no explanation and no persistent
        // notification to point back to.
        val callback = object : MediaProjection.Callback() {
            override fun onStop() {
                Log.w(TAG, "MediaProjection.onStop(): capture permission ended (system-revoked, not our own ACTION_STOP) - stopping overlay")
                Toast.makeText(
                    this@OverlayCaptureService,
                    getString(R.string.overlay_projection_stopped_toast),
                    Toast.LENGTH_LONG,
                ).show()
                stopSelf()
            }
        }
        projectionCallback = callback
        projection.registerCallback(callback, Handler(Looper.getMainLooper()))
        createCaptureDisplay(projection)
        return true
    }

    private fun stopProjection() {
        projectionCallback?.let { mediaProjection?.unregisterCallback(it) }
        projectionCallback = null
        mediaProjection?.stop()
        mediaProjection = null
    }

    /** The one display per grant, created with no surface: nothing is mirrored until a tap attaches the reader. */
    private fun createCaptureDisplay(projection: MediaProjection) {
        Log.d(TAG, "createCaptureDisplay (detached) at ${System.currentTimeMillis()}")
        val reader = ImageReader.newInstance(screenWidth, screenHeight, PixelFormat.RGBA_8888, 2)
        val display = projection.createVirtualDisplay(
            "GennedOverlayCapture",
            screenWidth,
            screenHeight,
            screenDensity,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            null,
            null,
            null,
        )
        captureReader = reader
        captureDisplay = display
    }

    private fun closeCaptureSurface() {
        val reader = captureReader ?: return
        val display = captureDisplay
        Log.d(TAG, "closeCaptureSurface at ${System.currentTimeMillis()}")
        display?.release()
        reader.close()
        captureReader = null
        captureDisplay = null
    }

    // --- Bubble window ---

    private fun addBubble() {
        val bubble = BubbleView(this)
        bubble.visibility = android.view.View.GONE // shown only over target apps, see watchForegroundApp()
        bubble.contentDescription = getString(R.string.bubble_cd_idle)
        bubbleView = bubble

        val preferences = (application as GennedApplication).container.preferences
        val size = (BUBBLE_SIZE_DP * resources.displayMetrics.density).toInt()
        // Where it was left last time, pulled back on screen if the screen changed since.
        val (startAtX, startAtY) = (preferences.bubblePosition ?: (0 to screenHeight / 3)).let { (x, y) ->
            BubbleGeometry.clampToScreen(x, y, size, screenWidth, screenHeight)
        }
        val params = WindowManager.LayoutParams(
            size,
            size,
            overlayWindowType(),
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = startAtX
            y = startAtY
        }
        layoutParams = params

        var downX = 0f
        var downY = 0f
        var startX = 0
        var startY = 0
        var moved = false

        // Taps go through performClick so a screen reader's double-tap works too.
        bubble.setOnClickListener { onBubbleTapped() }
        bubble.setOnTouchListener { view, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX
                    downY = event.rawY
                    startX = params.x
                    startY = params.y
                    moved = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (!moved && (abs(dx) > TOUCH_SLOP || abs(dy) > TOUCH_SLOP)) {
                        moved = true
                        showDismissTarget()
                    }
                    params.x = startX + dx.roundToInt()
                    params.y = startY + dy.roundToInt()
                    windowManager.updateViewLayout(bubble, params)
                    dismissTarget?.armed = isOverDismissTarget(event.rawX, event.rawY)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    val droppedOnDismiss = moved && event.action == MotionEvent.ACTION_UP &&
                        isOverDismissTarget(event.rawX, event.rawY)
                    hideDismissTarget()
                    when {
                        droppedOnDismiss -> {
                            Log.i(TAG, "Bubble dropped on the dismiss target: stopping overlay")
                            Toast.makeText(this, getString(R.string.overlay_dismissed_toast), Toast.LENGTH_LONG).show()
                            stopSelf()
                        }
                        moved -> {
                            val (x, y) = BubbleGeometry.clampToScreen(params.x, params.y, size, screenWidth, screenHeight)
                            params.x = x
                            params.y = y
                            runCatching { windowManager.updateViewLayout(bubble, params) }
                            preferences.bubblePosition = x to y
                        }
                        event.action == MotionEvent.ACTION_UP -> {
                            view.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
                            view.performClick()
                        }
                    }
                    true
                }
                else -> false
            }
        }

        windowManager.addView(bubble, params)
    }

    // --- Drag the bubble onto the "✕" at the bottom of the screen to turn the overlay off ---

    private fun dismissTargetSizePx(): Int = (DISMISS_TARGET_SIZE_DP * resources.displayMetrics.density).toInt()

    private fun dismissTargetBottomMarginPx(): Int =
        screenBottomInset + (DISMISS_TARGET_MARGIN_DP * resources.displayMetrics.density).toInt()

    private fun isOverDismissTarget(fingerX: Float, fingerY: Float): Boolean {
        val size = dismissTargetSizePx()
        return BubbleGeometry.isOverDismissTarget(
            fingerX = fingerX,
            fingerY = fingerY,
            targetCenterX = screenWidth / 2f,
            targetCenterY = screenHeight - dismissTargetBottomMarginPx() - size / 2f,
            radius = size * DISMISS_TARGET_REACH,
        )
    }

    private fun showDismissTarget() {
        if (dismissTarget != null) return
        val target = DismissTargetView(this)
        val size = dismissTargetSizePx()
        val params = WindowManager.LayoutParams(
            size,
            size,
            overlayWindowType(),
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = (screenWidth - size) / 2
            y = screenHeight - dismissTargetBottomMarginPx() - size
        }
        runCatching { windowManager.addView(target, params) }.onSuccess { dismissTarget = target }
    }

    private fun hideDismissTarget() {
        dismissTarget?.let { runCatching { windowManager.removeView(it) } }
        dismissTarget = null
    }

    private fun overlayWindowType(): Int = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
    } else {
        @Suppress("DEPRECATION")
        WindowManager.LayoutParams.TYPE_PHONE
    }

    private fun removeBubble() {
        hideDismissTarget()
        bubbleView?.let { runCatching { windowManager.removeView(it) } }
        bubbleView = null
    }

    // --- Capture + analyze ---

    private fun onBubbleTapped() {
        val bubble = bubbleView ?: return

        if (bubble.state == BubbleState.RESULT) {
            lastAnalysisId?.let { openResult(it) }
            return
        }
        if (analysisJob?.isActive == true) return

        val reader = captureReader ?: return

        analysisJob = serviceScope.launch {
            val capturedFile = captureGate.withAttached {
                // Mirror connected (just for this capture). Hiding the bubble keeps it out
                // of the frame and changes the screen, so a fresh frame is composed into
                // the just-attached surface even if nothing else on screen is moving.
                bubble.visibility = View.INVISIBLE
                try {
                    // Drop any frame queued while the bubble was still drawn.
                    runCatching { reader.acquireLatestImage()?.close() }
                    delay(CAPTURE_SETTLE_MS)
                    runCatching { captureFrame(reader) }.getOrNull()
                } finally {
                    // watchForegroundApp may have set GONE meanwhile; don't override that.
                    if (bubble.visibility == View.INVISIBLE) bubble.visibility = View.VISIBLE
                }
            }
            bubble.state = BubbleState.ANALYZING
            bubble.contentDescription = getString(R.string.bubble_cd_checking)
            bubble.startAnalyzingAnimation()

            if (capturedFile == null) {
                showTransientError(bubble)
                return@launch
            }

            // finally, so the screen capture is removed even if the service stops mid-analysis.
            val outcome = try {
                val cropSize = captureCrop()[2]
                val container = (application as GennedApplication).container
                val input = AnalysisInput(
                    originalFilePath = capturedFile.absolutePath,
                    normalizedFilePath = capturedFile.absolutePath,
                    originalMimeType = "image/jpeg",
                    widthPx = cropSize,
                    heightPx = cropSize,
                    fileSizeBytes = capturedFile.length(),
                )
                runCatching { container.analyzeImageUseCase.run(input, capturedFile, MediaKind.SCREEN) { } }
            } finally {
                capturedFile.delete()
            }
            val (analysisId, result) = outcome.getOrNull() ?: run {
                showTransientError(bubble)
                return@launch
            }

            bubble.stopAnalyzingAnimation()
            lastAnalysisId = analysisId
            val percent = displayPercent(result.aiLikelihood)
            bubble.resultPercent = percent
            bubble.resultColor = colorFor(result.classification)
            bubble.resultWord = getString(wordFor(result.classification))
            bubble.state = BubbleState.RESULT

            // Spoken for screen readers, and kept in the notification after the bubble resets.
            val summary = getString(R.string.bubble_result_summary, percent, getString(labelFor(result.classification)))
            bubble.contentDescription = getString(R.string.bubble_cd_result, summary)
            bubble.announceForAccessibility(summary)
            showLastCheckInNotification(summary, analysisId)

            delay(RESULT_DISPLAY_MS)
            if (bubble.state == BubbleState.RESULT) {
                bubble.state = BubbleState.IDLE
                bubble.contentDescription = getString(R.string.bubble_cd_idle)
            }
        }
    }

    private suspend fun showTransientError(bubble: BubbleView) {
        bubble.stopAnalyzingAnimation()
        bubble.state = BubbleState.ERROR
        val message = getString(R.string.bubble_cd_error)
        bubble.contentDescription = message
        bubble.announceForAccessibility(message)
        delay(ERROR_DISPLAY_MS)
        if (bubble.state == BubbleState.ERROR) {
            bubble.state = BubbleState.IDLE
            bubble.contentDescription = getString(R.string.bubble_cd_idle)
        }
    }

    private suspend fun captureFrame(reader: ImageReader): File? = withContext(Dispatchers.IO) {
        // A just-attached surface needs a frame or two to arrive; briefly poll rather
        // than failing on the first miss. The surface itself is NOT managed here - see
        // captureGate in onBubbleTapped.
        var image: Image? = null
        for (attempt in 0 until FRAME_POLL_ATTEMPTS) {
            image = reader.acquireLatestImage()
            if (image != null) break
            Thread.sleep(FRAME_POLL_DELAY_MS)
        }
        val img = image ?: return@withContext null

        try {
            val plane = img.planes[0]
            val buffer: ByteBuffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * screenWidth

            val rawBitmap = Bitmap.createBitmap(
                screenWidth + rowPadding / pixelStride,
                screenHeight,
                Bitmap.Config.ARGB_8888,
            )
            rawBitmap.copyPixelsFromBuffer(buffer)
            val (left, top, size) = captureCrop()
            val cropped = Bitmap.createBitmap(rawBitmap, left, top, size, size)
            if (cropped !== rawBitmap) rawBitmap.recycle()

            val outDir = File(cacheDir, "shared").apply { mkdirs() }
            val outFile = File(outDir, "overlay_capture_${System.currentTimeMillis()}.jpg")
            FileOutputStream(outFile).use { out -> cropped.compress(Bitmap.CompressFormat.JPEG, 92, out) }
            cropped.recycle()
            outFile
        } finally {
            img.close()
        }
    }

    private fun captureCrop(): IntArray =
        CaptureCrop.contentSquare(screenWidth, screenHeight, screenTopInset, screenBottomInset)

    private fun colorFor(classification: Classification): Int = when (classification) {
        Classification.HIGH -> Color.parseColor("#D8342A")
        Classification.UNCERTAIN -> Color.parseColor("#B5860B")
        Classification.LOW -> Color.parseColor("#2E7D4F")
    }

    private fun wordFor(classification: Classification): Int = when (classification) {
        Classification.HIGH -> R.string.bubble_word_high
        Classification.UNCERTAIN -> R.string.bubble_word_uncertain
        Classification.LOW -> R.string.bubble_word_low
    }

    private fun labelFor(classification: Classification): Int = when (classification) {
        Classification.HIGH -> R.string.result_classification_high
        Classification.UNCERTAIN -> R.string.result_classification_uncertain
        Classification.LOW -> R.string.result_classification_low
    }

    /** The bubble shows a result only briefly; the notification keeps the last one, and opens it when tapped. */
    private fun showLastCheckInNotification(summary: String, analysisId: Long) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        // Shown only if notifications are allowed (Android 13+ asks; see OverlaySetupScreen).
        runCatching { manager.notify(NOTIFICATION_ID, buildNotification(summary, analysisId)) }
    }

    private fun openResult(analysisId: Long) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK
            putExtra(Routes.EXTRA_OPEN_ANALYSIS_ID, analysisId)
        }
        startActivity(intent)
    }

    private fun buildNotification(lastCheck: String? = null, lastAnalysisId: Long? = null): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, getString(R.string.overlay_notification_channel), NotificationManager.IMPORTANCE_LOW),
            )
        }
        val stopIntent = PendingIntent.getService(
            this,
            0,
            Intent(this, OverlayCaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.overlay_notification_title))
            .setContentText(
                if (lastCheck != null) {
                    getString(R.string.overlay_notification_last_check, lastCheck)
                } else {
                    getString(R.string.overlay_notification_body)
                },
            )
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(0, getString(R.string.overlay_notification_stop), stopIntent)
        if (lastAnalysisId != null) {
            val openIntent = Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
                putExtra(Routes.EXTRA_OPEN_ANALYSIS_ID, lastAnalysisId)
            }
            builder.setContentIntent(
                PendingIntent.getActivity(
                    this,
                    1,
                    openIntent,
                    PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
                ),
            )
        }
        return builder.build()
    }

    companion object {
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_RESULT_DATA = "resultData"
        const val ACTION_STOP = "com.genned.app.overlay.STOP"

        private const val CHANNEL_ID = "overlay_capture"
        private const val NOTIFICATION_ID = 4201
        private const val TOUCH_SLOP = 12
        private const val RESULT_DISPLAY_MS = 10_000L
        private const val ERROR_DISPLAY_MS = 4_000L
        private const val BUBBLE_SIZE_DP = 56
        private const val DISMISS_TARGET_SIZE_DP = 64
        private const val DISMISS_TARGET_MARGIN_DP = 48

        /** How close to the target's centre a drop counts, as a multiple of the target's size. */
        private const val DISMISS_TARGET_REACH = 1.1f
        private const val FRAME_POLL_ATTEMPTS = 15
        private const val FRAME_POLL_DELAY_MS = 80L
        private const val CAPTURE_SETTLE_MS = 120L
        private const val TAG = "OverlayCaptureService"

        private val _running = MutableStateFlow(false)

        /** Whether the overlay session is live, so Settings can reflect it after being reopened. */
        val running: StateFlow<Boolean> = _running.asStateFlow()

        fun startIntent(context: Context, resultCode: Int, data: Intent): Intent =
            Intent(context, OverlayCaptureService::class.java).apply {
                putExtra(EXTRA_RESULT_CODE, resultCode)
                putExtra(EXTRA_RESULT_DATA, data)
            }
    }
}
