pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // c2pa-android (Content Credentials) is published only on GitHub Packages, which needs
        // a token; JitPack builds the same tagged source without one. Limited to that one group.
        maven("https://jitpack.io") {
            content { includeGroup("com.github.contentauth") }
        }
    }
}

rootProject.name = "genned"

include(":app")
include(":domain")
