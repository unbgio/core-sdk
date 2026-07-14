// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "UNBG",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "UNBG", targets: ["UNBG"])
    ],
    targets: [
        .binaryTarget(
            name: "unbgFFI",
            path: "dist/UNBG.xcframework"
        ),
        .target(
            name: "UNBG",
            dependencies: ["unbgFFI"],
            path: "Sources/UNBG"
        )
    ]
)
