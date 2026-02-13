# Security and Release Hygiene

This document defines release hygiene requirements for production artifacts.

## Required controls

- Dependency vulnerability scan via `cargo audit`.
- SBOM generation for the workspace (`cargo cyclonedx`).
- SHA-256 artifact manifest generation (`scripts/release-manifest.sh`).
- Artifact signatures when `UNBG_SIGNING_KEY_B64` is configured (`scripts/sign-artifacts.sh`).
- Build provenance attestations in release workflows.

## CI workflows

- Security checks: `.github/workflows/security.yml`
- Release metadata and provenance: `.github/workflows/release.yml`
- Platform artifact workflows:
  - `.github/workflows/android.yml`
  - `.github/workflows/ios.yml`
  - `.github/workflows/tauri-plugin.yml`

## Secret handling

- `UNBG_SIGNING_KEY_B64` should be stored as a repository or environment secret.
- Never commit private keys into the repository.

## Verification

Consumers should verify:

1. Artifact SHA-256 values from the release manifest.
2. Signature files (`*.sig`) when published.
3. Provenance attestations attached to release artifacts.
