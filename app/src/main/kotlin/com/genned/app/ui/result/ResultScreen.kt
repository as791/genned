package com.genned.app.ui.result

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Verified
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.genned.app.R
import com.genned.app.data.sharing.ShareIntentFactory
import com.genned.app.data.storage.MediaKind
import com.genned.app.data.storage.SavedAnalysis
import com.genned.app.text.displayPercent
import com.genned.app.ui.components.DisclaimerCard
import com.genned.app.ui.components.SectionHeading
import com.genned.app.ui.components.SignalCard
import com.genned.app.ui.components.classificationColor
import com.genned.app.ui.components.classificationLabel
import com.genned.domain.model.Classification
import com.genned.domain.model.DetectionSignal
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

// The phone's own date and time style and language.
private val timeFormatter = DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM, FormatStyle.SHORT)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    onCheckAnother: () -> Unit,
    onBack: () -> Unit,
    onOpenGuide: () -> Unit,
    viewModel: ResultViewModel = viewModel(factory = ResultViewModel.Factory),
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var showDeleteConfirm by remember { mutableStateOf(false) }
    val loaded = (uiState as? ResultUiState.Loaded)?.analysis

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    if (loaded != null) {
                        Column {
                            mediaKindLabel(loaded.mediaKind)?.let {
                                Text(text = stringResource(it), style = MaterialTheme.typography.titleMedium)
                            }
                            Text(
                                text = timeFormatter.format(
                                    Instant.ofEpochMilli(loaded.timestampMillis).atZone(ZoneId.systemDefault()),
                                ),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.cd_back))
                    }
                },
                actions = {
                    if (loaded != null) {
                        IconButton(onClick = { showDeleteConfirm = true }) {
                            Icon(Icons.Default.Delete, contentDescription = stringResource(R.string.result_delete))
                        }
                    }
                },
            )
        },
    ) { padding ->
        when (val state = uiState) {
            ResultUiState.Loading -> Column(
                modifier = Modifier.fillMaxSize().padding(padding),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                CircularProgressIndicator()
            }
            ResultUiState.NotFound -> ResultNotFound(onBack = onBack, contentPadding = padding)
            is ResultUiState.Loaded -> ResultContent(
                analysis = state.analysis,
                onShare = {
                    scope.launch {
                        val card = viewModel.renderShareCard() ?: return@launch
                        context.startActivity(ShareIntentFactory.forResultCard(context, card))
                    }
                },
                onCheckAnother = onCheckAnother,
                onOpenGuide = onOpenGuide,
                contentPadding = padding,
            )
        }
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text(text = stringResource(R.string.result_delete_confirm_title)) },
            text = { Text(text = stringResource(R.string.result_delete_confirm_body)) },
            confirmButton = {
                TextButton(onClick = {
                    showDeleteConfirm = false
                    viewModel.delete()
                    onBack()
                }) {
                    Text(text = stringResource(R.string.delete))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) {
                    Text(text = stringResource(R.string.cancel))
                }
            },
        )
    }
}

private fun mediaKindLabel(kind: MediaKind?): Int? = when (kind) {
    MediaKind.PHOTO -> R.string.result_kind_photo
    MediaKind.VIDEO -> R.string.result_kind_video
    MediaKind.SCREEN -> R.string.result_kind_screen
    null -> null
}

@Composable
private fun ResultNotFound(onBack: () -> Unit, contentPadding: PaddingValues) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = stringResource(R.string.result_not_found_title),
            style = MaterialTheme.typography.titleLarge,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.result_not_found_body),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onBack) {
            Text(text = stringResource(R.string.cd_back))
        }
    }
}

@Composable
private fun ResultContent(
    analysis: SavedAnalysis,
    onShare: () -> Unit,
    onCheckAnother: () -> Unit,
    onOpenGuide: () -> Unit,
    contentPadding: PaddingValues,
) {
    val result = analysis.result
    val verified = result.verifiedCredentials
    val arranged = remember(result.signals) { ResultPresentation.arrange(result.signals) }
    var showNotAvailable by rememberSaveable { mutableStateOf(false) }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 24.dp),
        contentPadding = PaddingValues(top = 8.dp, bottom = 32.dp),
    ) {
        item {
            ResultHeader(analysis, verified)
            TextButton(onClick = onOpenGuide, contentPadding = PaddingValues(0.dp)) {
                Text(text = stringResource(R.string.result_what_does_this_mean))
            }
            Spacer(modifier = Modifier.height(12.dp))
        }
        item {
            SectionHeading(stringResource(R.string.result_why_heading))
        }
        items(arranged.shown) { signal ->
            SignalCard(signal = signal, modifier = Modifier.padding(bottom = 12.dp))
        }
        if (arranged.notAvailable.isNotEmpty()) {
            item {
                NotAvailableRow(
                    count = arranged.notAvailable.size,
                    expanded = showNotAvailable,
                    onToggle = { showNotAvailable = !showNotAvailable },
                )
            }
            if (showNotAvailable) {
                items(arranged.notAvailable) { signal ->
                    SignalCard(signal = signal, modifier = Modifier.padding(bottom = 12.dp))
                }
            }
        }
        item {
            Spacer(modifier = Modifier.height(12.dp))
            SectionHeading(stringResource(R.string.result_important_heading))
            DisclaimerCard(
                text = stringResource(
                    if (verified != null) R.string.result_verified_disclaimer else R.string.result_disclaimer,
                ),
            )
            // The first limitation is the general "this is an estimate" note, shown in the card above.
            result.limitations.drop(1).forEach { limitation ->
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = limitation,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(modifier = Modifier.height(24.dp))
            Button(onClick = onCheckAnother, modifier = Modifier.fillMaxWidth()) {
                Text(text = stringResource(R.string.result_check_another))
            }
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedButton(onClick = onShare, modifier = Modifier.fillMaxWidth()) {
                Text(text = stringResource(R.string.result_share))
            }
        }
    }
}

/** One row standing in for every check that had nothing to report; tap to see them. */
@Composable
private fun NotAvailableRow(count: Int, expanded: Boolean, onToggle: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .clickable(
                onClickLabel = stringResource(
                    if (expanded) R.string.result_hide_not_available else R.string.result_show_not_available,
                ),
                role = Role.Button,
                onClick = onToggle,
            )
            .padding(vertical = 12.dp, horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = pluralStringResource(R.plurals.result_checks_not_available, count, count),
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        Icon(
            imageVector = if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * The answer in plain words. A normal result is an estimate: verdict, percentage, and what
 * that means. A result decided by verified Content Credentials is not an estimate, so it
 * shows a "Verified" badge and what the signer declared instead of a percentage.
 */
@Composable
private fun ResultHeader(analysis: SavedAnalysis, verified: DetectionSignal?) {
    val result = analysis.result
    val bitmap = remember(analysis.thumbnailPath) {
        analysis.thumbnailPath?.let { path ->
            runCatching { BitmapFactory.decodeFile(path)?.asImageBitmap() }.getOrNull()
        }
    }
    val accent = classificationColor(result.classification)

    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap,
                    contentDescription = stringResource(R.string.cd_image_preview),
                    contentScale = ContentScale.Crop,
                    modifier = Modifier
                        .size(96.dp)
                        .clip(RoundedCornerShape(16.dp)),
                )
                Spacer(modifier = Modifier.width(16.dp))
            }
            Column {
                if (verified != null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.Verified,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = stringResource(R.string.result_verified_badge),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = stringResource(
                            if (result.classification == Classification.LOW) {
                                R.string.result_verified_capture
                            } else {
                                R.string.result_verified_ai
                            },
                        ),
                        style = MaterialTheme.typography.headlineMedium,
                        color = accent,
                    )
                } else {
                    Text(
                        text = classificationLabel(result.classification),
                        style = MaterialTheme.typography.headlineMedium,
                        color = accent,
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = stringResource(R.string.result_ai_likelihood_percent, displayPercent(result.aiLikelihood)),
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = verified?.description ?: stringResource(meaningOf(result.classification)),
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

private fun meaningOf(classification: Classification): Int = when (classification) {
    Classification.HIGH -> R.string.result_meaning_high
    Classification.UNCERTAIN -> R.string.result_meaning_uncertain
    Classification.LOW -> R.string.result_meaning_low
}
