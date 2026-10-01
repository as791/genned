# ONNX Runtime uses JNI; keep its Java bindings so native method signatures resolve.
-keep class ai.onnxruntime.** { *; }

# Room-generated code and kotlinx.serialization use reflection-adjacent codegen that
# is already annotation-processed at compile time; standard AGP consumer rules from
# each library cover the rest. Nothing app-specific is obfuscation-sensitive here
# because DTOs are all plain data classes with no reflection-based (de)serialization
# outside kotlinx.serialization, which ships its own consumer ProGuard rules.

# Release builds carry no debug-level logging: R8 removes Log.d / Log.v calls (and the
# string building feeding them). Log.i/w/e stay; per internal-docs/PRIVACY.md "Logging"
# none of them ever contain image bytes, metadata content or file names.
-assumenosideeffects class android.util.Log {
    public static int d(...);
    public static int v(...);
}
