package com.genned.app.ui.guide

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.unit.dp
import com.genned.app.R
import com.genned.app.ui.components.SectionHeading
import com.genned.app.ui.components.classificationColor
import com.genned.app.ui.components.classificationLabel
import com.genned.domain.model.Classification

/** How to check something from another app, and how to read the three outcomes. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GuideScreen(onBack: () -> Unit) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(R.string.guide_title)) },
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
            SectionHeading(stringResource(R.string.guide_share_heading))
            ShareSteps()
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = stringResource(R.string.guide_share_note),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(24.dp))

            SectionHeading(stringResource(R.string.guide_results_heading))
            Outcome(
                title = classificationLabel(Classification.HIGH),
                color = classificationColor(Classification.HIGH),
                body = stringResource(R.string.guide_result_high),
            )
            Outcome(
                title = classificationLabel(Classification.UNCERTAIN),
                color = classificationColor(Classification.UNCERTAIN),
                body = stringResource(R.string.guide_result_uncertain),
            )
            Outcome(
                title = classificationLabel(Classification.LOW),
                color = classificationColor(Classification.LOW),
                body = stringResource(R.string.guide_result_low),
            )
            Outcome(
                title = stringResource(R.string.result_verified_badge),
                color = MaterialTheme.colorScheme.primary,
                body = stringResource(R.string.guide_result_verified),
            )
            Spacer(modifier = Modifier.height(16.dp))

            SectionHeading(stringResource(R.string.guide_tips_heading))
            Text(text = stringResource(R.string.guide_tip_original), style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(R.string.guide_tip_video), style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = stringResource(R.string.guide_tip_private), style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

/** The three steps for checking something from another app. Also shown on Home before the first check. */
@Composable
fun ShareSteps(modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Step(1, stringResource(R.string.guide_step_1))
        Step(2, stringResource(R.string.guide_step_2))
        Step(3, stringResource(R.string.guide_step_3))
    }
}

@Composable
private fun Step(number: Int, text: String) {
    Row(modifier = Modifier.padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(28.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primaryContainer)
                // The number is decoration; the step text is read on its own.
                .clearAndSetSemantics { },
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = number.toString(),
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )
        }
        Spacer(modifier = Modifier.width(12.dp))
        Text(text = text, style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
private fun Outcome(title: String, color: Color, body: String) {
    Column(modifier = Modifier.padding(bottom = 12.dp)) {
        Text(text = title, style = MaterialTheme.typography.titleMedium, color = color)
        Text(
            text = body,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
