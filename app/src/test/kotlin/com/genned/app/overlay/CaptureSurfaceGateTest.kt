package com.genned.app.overlay

import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertThrows
import org.junit.Test

/** The battery guard: the screen mirror is connected only during a capture. */
class CaptureSurfaceGateTest {

    private val calls = mutableListOf<String>()
    private val gate = CaptureSurfaceGate(attach = { calls += "attach" }, detach = { calls += "detach" })

    @Test
    fun `disconnected until a capture, connected only inside it`() = runTest {
        assertThat(gate.attached).isFalse()
        assertThat(calls).isEmpty()
        val seenInside = gate.withAttached { gate.attached }
        assertThat(seenInside).isTrue()
        assertThat(gate.attached).isFalse()
        assertThat(calls).containsExactly("attach", "detach").inOrder()
    }

    @Test
    fun `a failed capture still disconnects`() = runTest {
        assertThrows(IllegalStateException::class.java) {
            kotlinx.coroutines.runBlocking { gate.withAttached { error("capture failed") } }
        }
        assertThat(gate.attached).isFalse()
        assertThat(calls).containsExactly("attach", "detach").inOrder()
    }

    @Test
    fun `a cancelled capture still disconnects`() = runTest {
        val started = CompletableDeferred<Unit>()
        val job = launch {
            gate.withAttached {
                started.complete(Unit)
                CompletableDeferred<Unit>().await() // never completes
            }
        }
        started.await()
        assertThat(gate.attached).isTrue()
        job.cancel(CancellationException("overlay stopped"))
        job.join()
        assertThat(gate.attached).isFalse()
        assertThat(calls).containsExactly("attach", "detach").inOrder()
    }

    @Test
    fun `a nested capture does not connect twice or disconnect early`() = runTest {
        gate.withAttached {
            gate.withAttached { }
            assertThat(gate.attached).isTrue()
        }
        assertThat(calls).containsExactly("attach", "detach").inOrder()
    }
}
