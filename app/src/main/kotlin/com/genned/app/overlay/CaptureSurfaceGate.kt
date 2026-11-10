package com.genned.app.overlay

/**
 * Connects the screen mirror only while a capture is running. While it is connected,
 * Android copies every frame of the screen into the capture buffer, which costs battery.
 * So it stays disconnected the rest of the time, including while you scroll Instagram or
 * WhatsApp, and is connected only for the moment of a tap.
 *
 * [attach]/[detach] switch the one VirtualDisplay's surface (see OverlayCaptureService:
 * the display itself can be created only once per capture grant). Main thread only.
 */
internal class CaptureSurfaceGate(
    private val attach: () -> Unit,
    private val detach: () -> Unit,
) {
    var attached: Boolean = false
        private set

    /** Runs [block] with the mirror connected, and always disconnects it after, also on failure or cancellation. */
    suspend fun <T> withAttached(block: suspend () -> T): T {
        if (attached) return block()
        attach()
        attached = true
        try {
            return block()
        } finally {
            detach()
            attached = false
        }
    }
}
