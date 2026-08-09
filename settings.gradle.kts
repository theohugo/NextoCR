// Modified by NextoCR from the Apache-2.0 crforge build configuration.
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}

rootProject.name = "nextocr"

include("core")
include("desktop")
include("gym-bridge")
include("data")
