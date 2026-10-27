plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.dagger.hilt.android")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.onndetect"
    compileSdk = 36
    ndkVersion = "26.1.10909125"

    defaultConfig {
        applicationId = "com.onndetect"
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "1.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20"
            }
        }
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    signingConfigs {
        create("release") {
            val storeFilePath = System.getenv("ONN_RELEASE_STORE_FILE")
                ?: (project.findProperty("ONN_RELEASE_STORE_FILE") as? String)
            if (storeFilePath != null && file(storeFilePath).exists()) {
                storeFile = file(storeFilePath)
                storePassword = System.getenv("ONN_RELEASE_STORE_PASSWORD")
                    ?: (project.findProperty("ONN_RELEASE_STORE_PASSWORD") as? String)
                keyAlias = System.getenv("ONN_RELEASE_KEY_ALIAS")
                    ?: (project.findProperty("ONN_RELEASE_KEY_ALIAS") as? String)
                keyPassword = System.getenv("ONN_RELEASE_KEY_PASSWORD")
                    ?: (project.findProperty("ONN_RELEASE_KEY_PASSWORD") as? String)
            } else {
                // Non-production environment fallback: mirror debug keystore for local/CI release verification
                val debugConfig = signingConfigs.getByName("debug")
                storeFile = debugConfig.storeFile
                storePassword = debugConfig.storePassword
                keyAlias = debugConfig.keyAlias
                keyPassword = debugConfig.keyPassword
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    
    
    buildFeatures {
        compose = true
        buildConfig = true
    }
    
    
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    
    androidResources {
        noCompress += listOf("tflite", "task")
    }
}

dependencies {
    val composeBomVersion = "2025.09.01"
    val roomVersion = "2.8.5"

    // Core AndroidX
    implementation("androidx.core:core-ktx:1.17.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.4")
    implementation("androidx.activity:activity-compose:1.11.0")

    // Jetpack Compose
    implementation(platform("androidx.compose:compose-bom:$composeBomVersion"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.navigation:navigation-compose:2.9.4")

    // Hilt (Dependency Injection)
    implementation("com.google.dagger:hilt-android:2.57.2")
    ksp("com.google.dagger:hilt-android-compiler:2.57.2")
    implementation("androidx.hilt:hilt-lifecycle-viewmodel-compose:1.3.0")

    // Room (Local Database)
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    // Standalone, bundled CPU runtime: no Play-services download required.
    implementation("com.google.ai.edge.litert:litert:1.4.2")

    // MediaPipe FaceLandmarker (F5): 478-point face mesh for real perioral
    // landmarks; ~3.8 MB float16 .task asset bundled, GPU with CPU fallback.
    implementation("com.google.mediapipe:tasks-vision:0.10.35")

    // EXIF
    implementation("androidx.exifinterface:exifinterface:1.4.2")

    // Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20260814")
    // Test-only X.509 fixture generation for the C2PA suite (runtime keys,
    // never shipped). Production code has no BC dependency.
    testImplementation("org.bouncycastle:bcprov-jdk18on:1.80")
    testImplementation("org.bouncycastle:bcpkix-jdk18on:1.80")
    androidTestImplementation("androidx.room:room-testing:$roomVersion")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")
    androidTestImplementation(platform("androidx.compose:compose-bom:$composeBomVersion"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

ksp { arg("room.schemaLocation", "$projectDir/schemas") }

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
