# Proguard rules for ONN-DETECT

# Suppress AutoValue annotation processor classes bundled transitively in MediaPipe tasks-vision
-dontwarn javax.annotation.processing.**
-dontwarn javax.lang.model.**
-dontwarn com.google.auto.value.**

# Keep MediaPipe and TFLite classes required at runtime
-keep class com.google.mediapipe.tasks.** { *; }
-keep class org.tensorflow.lite.** { *; }
