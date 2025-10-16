# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.3] - 2025-10-15

### Added

- Python 3.12 and 3.13 support
- Development requirements in requirements.dev

### Changed

- Updated docstrings throughout the codebase
- Auto-formatted code for consistency
- Migrated from setup.cfg to pyproject.toml
- Removed 760m/s boundary on mu estimation formula when generating G/Gmax
  curve parameters for the hybrid hyperbolic model

### Removed

- Python 3.8 support (minimum version is now 3.9)

### Fixed

- Python 3.12 pipeline issues
- GitHub Pages deployment workflow permissions by adding environment configuration
