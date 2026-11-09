package com.genned.app.data.detection.provenance

import android.content.res.AssetManager
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/**
 * Reader settings for Content Credentials verification: which certificate authorities count
 * as trusted signers, and never fetching anything from the network.
 *
 * The trust anchors are bundled in `assets/c2pa/` (see the README there for sources): the
 * official C2PA Trust List and the Content Authenticity Initiative's interim trust list,
 * which older signed content (before the C2PA list existed) chains to. A manifest that
 * verifies but chains to neither is reported as "signed by an unrecognized signer" and is
 * never used as proof. The lists are updated with app releases.
 */
object C2paTrustSettings {

    const val ASSET_DIR = "c2pa"
    private val ANCHOR_FILES = listOf("c2pa-trust-list.pem", "interim-trust-list.pem")
    private const val TRUST_CONFIG_FILE = "trust-config.cfg"

    fun fromAssets(assets: AssetManager): String {
        fun asset(name: String) = assets.open("$ASSET_DIR/$name").bufferedReader().use { it.readText() }
        return json(
            trustAnchorsPem = ANCHOR_FILES.joinToString("\n") { asset(it) },
            trustConfig = asset(TRUST_CONFIG_FILE),
        )
    }

    /** c2pa-rs settings JSON. [trustConfig] lists the certificate purposes (EKU OIDs) accepted for signing. */
    fun json(trustAnchorsPem: String, trustConfig: String?): String = buildJsonObject {
        putJsonObject("verify") {
            put("verify_trust", true)
            put("remote_manifest_fetch", false)
        }
        putJsonObject("trust") {
            put("trust_anchors", trustAnchorsPem)
            if (trustConfig != null) put("trust_config", trustConfig)
        }
    }.toString()
}
