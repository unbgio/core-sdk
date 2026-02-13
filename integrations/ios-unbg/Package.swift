// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "UNBG",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "UNBG", targets: ["UNBG", "UNBGGenerated"])
    ],
    targets: [
        .target(
            name: "UNBG",
            path: "Sources",
            publicHeadersPath: nil
        ),
        .target(
            name: "UNBGGenerated",
            path: "generated"
        )
    ]
)
