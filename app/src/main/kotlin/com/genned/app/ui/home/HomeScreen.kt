package com.genned.app.ui.home

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.HelpOutline
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.genned.app.R
import com.genned.app.ui.components.HistoryRow
import com.genned.app.ui.guide.ShareSteps

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onMediaSelected: (uri: String, isVideo: Boolean) -> Unit,
    onOpenHistory: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenGuide: () -> Unit,
    onOpenHistoryItem: (Long) -> Unit,
    viewModel: HomeViewModel = viewModel(factory = HomeViewModel.Factory),
) {
    val recentChecks by viewModel.recentChecks.collectAsState()

    val context = LocalContext.current
    val pickMedia = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let {
            val isVideo = context.contentResolver.getType(it)?.startsWith("video/") == true
            onMediaSelected(it.toString(), isVideo)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(R.string.app_name)) },
                actions = {
                    IconButton(onClick = onOpenGuide) {
                        Icon(Icons.AutoMirrored.Filled.HelpOutline, contentDescription = stringResource(R.string.cd_help_icon))
                    }
                    IconButton(onClick = onOpenHistory) {
                        Icon(Icons.Default.History, contentDescription = stringResource(R.string.cd_history_icon))
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Default.Settings, contentDescription = stringResource(R.string.cd_settings_icon))
                    }
                },
            )
        },
    ) { padding ->
        // One scrolling list, so everything stays reachable on small screens and in landscape.
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 24.dp),
            contentPadding = PaddingValues(bottom = 24.dp),
        ) {
            item {
                Text(
                    text = stringResource(R.string.home_title),
                    style = MaterialTheme.typography.headlineLarge,
                    modifier = Modifier.padding(top = 24.dp, bottom = 24.dp).semantics { heading() },
                )
                Button(
                    onClick = {
                        pickMedia.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo),
                        )
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(text = stringResource(R.string.home_choose_media))
                }
                Text(
                    text = stringResource(R.string.home_share_hint),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 12.dp, bottom = 24.dp),
                )
            }

            if (recentChecks.isEmpty()) {
                // Before the first check: show how sharing from another app works.
                item { GuideCard(onOpenGuide = onOpenGuide) }
            } else {
                item {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = stringResource(R.string.home_recent_checks),
                            style = MaterialTheme.typography.titleLarge,
                            modifier = Modifier.semantics { heading() },
                        )
                        TextButton(onClick = onOpenHistory) {
                            Text(text = stringResource(R.string.home_view_all_history))
                        }
                    }
                }
                items(recentChecks, key = { it.id }) { entry ->
                    HistoryRow(entry = entry, onClick = { onOpenHistoryItem(entry.id) })
                }
            }
        }
    }
}

@Composable
private fun GuideCard(onOpenGuide: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(R.string.home_guide_heading),
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.padding(bottom = 8.dp).semantics { heading() },
            )
            ShareSteps()
            TextButton(onClick = onOpenGuide) {
                Text(text = stringResource(R.string.home_guide_more))
            }
        }
    }
}
