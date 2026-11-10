package com.genned.app.ui.settings

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.genned.app.BuildConfig
import com.genned.app.R
import com.genned.app.data.detection.classifier.EnsembleConfig
import com.genned.app.data.detection.classifier.ModelAssets
import com.genned.app.data.detection.classifier.ModelConfig
import com.genned.app.debug.LogcatCapture
import com.genned.app.overlay.OverlayCaptureService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.genned.app.ui.components.SectionHeading

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    onOpenLicenses: () -> Unit,
    onOpenGuide: () -> Unit,
    onOpenOverlaySetup: () -> Unit,
) {
    val context = LocalContext.current
    val modelBundled = remember { ModelAssets.isBundled(context) }
    val ensembleBundled = remember {
        modelBundled && ModelAssets.isBundled(context, ModelAssets.COMMUNITY_FORENSICS_ASSET_PATH)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(R.string.settings_title)) },
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
            SectionHeading(stringResource(R.string.settings_privacy_heading))
            Text(text = stringResource(R.string.settings_privacy_body), style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedButton(onClick = { openPrivacyPolicy(context) }) {
                Text(text = stringResource(R.string.settings_privacy_policy))
            }
            Spacer(modifier = Modifier.height(24.dp))

            SectionHeading(stringResource(R.string.settings_model_heading))
            Text(
                text = if (modelBundled) {
                    stringResource(
                        R.string.settings_model_body_present,
                        if (ensembleBundled) EnsembleConfig.DISPLAY_NAME else ModelConfig.DISPLAY_NAME,
                    )
                } else {
                    stringResource(R.string.settings_model_body_absent)
                },
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(modifier = Modifier.height(24.dp))

            SectionHeading(stringResource(R.string.settings_help_heading))
            OutlinedButton(onClick = onOpenGuide) {
                Text(text = stringResource(R.string.home_guide_more))
            }
            Spacer(modifier = Modifier.height(24.dp))

            SectionHeading(stringResource(R.string.settings_about_heading))
            Text(
                text = stringResource(R.string.settings_version, BuildConfig.VERSION_NAME),
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedButton(onClick = onOpenLicenses) {
                Text(text = stringResource(R.string.settings_licenses))
            }
            Spacer(modifier = Modifier.height(24.dp))

            HorizontalDivider()
            Spacer(modifier = Modifier.height(24.dp))
            SectionHeading(stringResource(R.string.settings_experimental_heading))
            OverlaySection(onOpenOverlaySetup)
            // Diagnostics for testers only: release builds have neither this section nor the
            // debug-level log lines it would copy (R8 strips Log.d/Log.v, see proguard-rules.pro).
            if (BuildConfig.DEBUG) {
                Spacer(modifier = Modifier.height(24.dp))
                DebugLogSection()
            }
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

/** The bubble has its own setup screen (OverlaySetupScreen); Settings only links to it and shows whether it is on. */
@Composable
private fun OverlaySection(onOpenOverlaySetup: () -> Unit) {
    val overlayRunning by OverlayCaptureService.running.collectAsState()

    Text(text = stringResource(R.string.settings_overlay_body), style = MaterialTheme.typography.bodyMedium)
    Spacer(modifier = Modifier.height(8.dp))
    OutlinedButton(onClick = onOpenOverlaySetup) {
        Text(
            text = stringResource(
                if (overlayRunning) R.string.settings_overlay_manage else R.string.settings_overlay_open,
            ),
        )
    }
}

/**
 * Lets a user hand over real evidence for an overlay bug (bubble not showing
 * over some app, unexpected battery use) entirely from their phone, no
 * computer/adb needed — see [LogcatCapture] for why that's possible without
 * any special permission, and internal-docs/ARCHITECTURE.md "Screen overlay
 * (experimental)" for what's actually logged.
 */
@Composable
private fun DebugLogSection() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var confirmation by remember { mutableStateOf<String?>(null) }

    SectionHeading(stringResource(R.string.settings_debug_log_heading))
    Text(text = stringResource(R.string.settings_debug_log_body), style = MaterialTheme.typography.bodyMedium)
    Spacer(modifier = Modifier.height(12.dp))

    OutlinedButton(
        onClick = {
            scope.launch {
                val log = withContext(Dispatchers.IO) {
                    LogcatCapture.captureRecent(
                        listOf("ForegroundAppWatcher", "OverlayCaptureService", "AIImageClassifier"),
                    )
                }
                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clipboard.setPrimaryClip(ClipData.newPlainText("Genned debug log", log))
                confirmation = context.getString(
                    R.string.settings_debug_log_copied,
                    log.lineSequence().count(),
                )
            }
        },
    ) {
        Text(text = stringResource(R.string.settings_debug_log_copy))
    }

    confirmation?.let { message ->
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
    }
}

/**
 * Opens the public privacy policy (PRIVACY.md on GitHub) in the browser. The app itself has
 * no internet permission; the browser loads the page.
 */
private fun openPrivacyPolicy(context: Context) {
    val intent = Intent(Intent.ACTION_VIEW, Uri.parse(context.getString(R.string.settings_privacy_policy_url)))
    try {
        context.startActivity(intent)
    } catch (e: ActivityNotFoundException) {
        // No browser installed: nothing can open the link.
    }
}
