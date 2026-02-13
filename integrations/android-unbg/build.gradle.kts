plugins {
    id("com.android.library")
    kotlin("android")
    id("maven-publish")
}

android {
    namespace = "com.unbg.sdk"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }

    sourceSets["main"].jniLibs.srcDir("dist/aar/jni")
    sourceSets["main"].java.srcDir("generated")
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib:2.2.20")
}

val verifyGeneratedBindings by tasks.registering {
    doLast {
        val generated = file("generated")
        val hasKotlin = generated.walkTopDown().any { it.isFile && it.extension == "kt" }
        if (!hasKotlin) {
            throw GradleException("Missing generated Kotlin UniFFI bindings. Run scripts/build-android.sh first.")
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn(verifyGeneratedBindings)
}

publishing {
    publications {
        create<MavenPublication>("release") {
            groupId = "com.unbg"
            artifactId = "unbg-android"
            version = "0.1.0"
            afterEvaluate { from(components["release"]) }
        }
    }
    repositories {
        maven {
            name = "localUnbg"
            url = uri(layout.buildDirectory.dir("repo"))
        }
    }
}
