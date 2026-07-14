plugins {
    id("com.android.library") version "9.3.0"
    id("maven-publish")
}

android {
    namespace = "com.unbg.sdk"
    compileSdk = 37
    buildToolsVersion = "37.0.0"

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }

    publishing {
        singleVariant("release")
    }
}

androidComponents.onVariants { variant ->
    variant.sources.jniLibs?.addStaticSourceDirectory("dist/aar/jni")
    variant.sources.kotlin?.addStaticSourceDirectory("generated")
    variant.sources.resources?.addStaticSourceDirectory("generated-resources")
}

dependencies {
    api("net.java.dev.jna:jna:5.19.1@aar")
}

val verifyGeneratedBindings = tasks.register("verifyGeneratedBindings") {
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
