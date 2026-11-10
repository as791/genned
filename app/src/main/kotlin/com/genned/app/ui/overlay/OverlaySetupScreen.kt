package com.genned.app.ui.overlay

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import com.genned.app.R
import com.genned.app.overlay.OverlayCaptureService
import com.genned.app.overlay.OverlayPermissions
import com.genned.app.overlay.OverlaySetupState
import com.genned.app.overlay.OverlaySetupStep
import com.genned.app.ui.components.SectionHeading

/**
 * Turns the Instagram/WhatsApp bubble on. Android needs three separate grants, each given
 * in a different system screen. This shows them as a checklist with one button for the next
 * step, and when you come back from a system screen it moves on by itself, so you never
 * have to work out what to press again.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OverlaySetupScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val running by OverlayCaptureService.running.collectAsState()
    var canDraw by remember { mutableStateOf(OverlayPermissions.canDrawOverlays(context)) }
    var hasUsage by remember { mutableStateOf(OverlayPermissions.hasUsageAccess(context)) }
    var notificationsAllowed by remember { mutableStateOf(notificationsAllowed(context)) }
    val state = OverlaySetupState(canDrawOverApps = canDraw, hasUsageAccess = hasUsage, running = running)

    fun refresh() {
        canDraw = OverlayPermissions.canDrawOverlays(context)
        hasUsage = OverlayPermissions.hasUsageAccess(context)
        notificationsAllowed = notificationsAllowed(context)
    }
    // Permissions can also change while the user is away in system settings.
    LifecycleResumeEffect(Unit) {
        refresh()
        onPauseOrDispose { }
    }

    val projectionManager = remember {
        context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    }
    val projectionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        val data = result.data
        if (result.resultCode == Activity.RESULT_OK && data != null) {
            ContextCompat.startForegroundService(
                context,
                OverlayCaptureService.startIntent(context, result.resultCode, data),
            )
        }
    }

    // When the user comes back from a system screen having granted that step, the next
    // step starts by itself. If they came back without granting it, nothing reopens.
    var lastStarted by remember { mutableStateOf<OverlaySetupStep?>(null) }
    var autoStart by remember { mutableStateOf<OverlaySetupStep?>(null) }
    val settingsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) {
        refresh()
        val next = OverlaySetupState(
            canDrawOverApps = OverlayPermissions.canDrawOverlays(context),
            hasUsageAccess = OverlayPermissions.hasUsageAccess(context),
            running = running,
        ).nextStep
        if (next != null && next != lastStarted) autoStart = next
    }
    fun startStep(step: OverlaySetupStep) {
        lastStarted = step
        when (step) {
            OverlaySetupStep.DRAW_OVER_APPS -> settingsLauncher.launch(OverlayPermissions.overlayPermissionIntent(context))
            OverlaySetupStep.USAGE_ACCESS -> settingsLauncher.launch(OverlayPermissions.usageAccessSettingsIntent())
            OverlaySetupStep.SCREEN_CAPTURE -> projectionLauncher.launch(projectionManager.createScreenCaptureIntent())
        }
    }
    LaunchedEffect(autoStart) {
        autoStart?.let { step ->
            autoStart = null
            startStep(step)
        }
    }

    val notificationLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { refresh() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(R.string.overlay_setup_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.cd_back))
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 24.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(R.string.overlay_setup_intro), style = MaterialTheme.typography.bodyLarge)
            Spacer(modifier = Modifier.height(24.dp))

            if (running) {
                RunningCard(onStop = { context.stopService(Intent(context, OverlayCaptureService::class.java)) })
            } else {
                SectionHeading(stringResource(R.string.overlay_setup_steps_heading))
                StepRow(
                    number = 1,
                    title = stringResource(R.string.overlay_step_draw_title),
                    body = stringResource(R.string.overlay_step_draw_body),
                    done = state.isDone(OverlaySetupStep.DRAW_OVER_APPS),
                    isNext = state.nextStep == OverlaySetupStep.DRAW_OVER_APPS,
                    actionLabel = stringResource(R.string.overlay_step_action),
                    onAction = { startStep(OverlaySetupStep.DRAW_OVER_APPS) },
                )
                StepRow(
                    number = 2,
                    title = stringResource(R.string.overlay_step_usage_title),
                    body = stringResource(R.string.overlay_step_usage_body),
                    done = state.isDone(OverlaySetupStep.USAGE_ACCESS),
                    isNext = state.nextStep == OverlaySetupStep.USAGE_ACCESS,
                    actionLabel = stringResource(R.string.overlay_step_action),
                    onAction = { startStep(OverlaySetupStep.USAGE_ACCESS) },
                )
                StepRow(
                    number = 3,
                    title = stringResource(R.string.overlay_step_capture_title),
                    body = stringResource(R.string.overlay_step_capture_body),
                    done = false,
                    isNext = state.nextStep == OverlaySetupStep.SCREEN_CAPTURE,
                    actionLabel = stringResource(R.string.overlay_setup_start),
                    onAction = { startStep(OverlaySetupStep.SCREEN_CAPTURE) },
                )
            }

            if (!notificationsAllowed && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = stringResource(R.string.overlay_notifications_title),
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    text = stringResource(R.string.overlay_notifications_body),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(onClick = { notificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS) }) {
                    Text(text = stringResource(R.string.overlay_notifications_action))
                }
            }
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

private fun notificationsAllowed(context: Context): Boolean =
    Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
        ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED

@Composable
private fun RunningCard(onStop: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(R.string.overlay_setup_running_title),
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = stringResource(R.string.overlay_setup_running_body),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )
            Spacer(modifier = Modifier.height(12.dp))
            OutlinedButton(onClick = onStop) {
                Text(text = stringResource(R.string.overlay_setup_stop))
            }
        }
    }
}

@Composable
private fun StepRow(
    number: Int,
    title: String,
    body: String,
    done: Boolean,
    isNext: Boolean,
    actionLabel: String,
    onAction: () -> Unit,
) {
    val doneLabel = stringResource(R.string.overlay_step_done)
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 10.dp)) {
        Box(
            modifier = Modifier
                .size(32.dp)
                .clip(CircleShape)
                .background(
                    if (done) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
                )
                .clearAndSetSemantics { },
            contentAlignment = Alignment.Center,
        ) {
            if (done) {
                Icon(
                    imageVector = Icons.Default.Check,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onPrimary,
                    modifier = Modifier.size(20.dp),
                )
            } else {
                Text(
                    text = number.toString(),
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Spacer(modifier = Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.semantics { if (done) stateDescription = doneLabel },
            )
            Text(
                text = body,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (isNext) {
                Spacer(modifier = Modifier.height(8.dp))
                Button(onClick = onAction) {
                    Text(text = actionLabel)
                }
            }
        }
    }
}
