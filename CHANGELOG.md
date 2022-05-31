# CHANGELOG

## [v0.3.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.3.0) - 2022-05-31 17:28:49

<!-- Release notes generated using configuration in .github/release.yml at v0.3.0  -->

## What's Changed
Feature ranges for `a`, `b`, `c`, `volume`, and `distance` are chosen based on Materials Project data for all structures with fewer than 52 sites.

### Other Changes
* ranges of features by @hasan-sayeed in https://github.com/sparks-baird/xtal2png/pull/39
* Colab feature ranges notebook updates by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/45
* matplotlibify by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/40
* rename to 2.0-materials-project-feature-ranges.ipynb by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/46
* refactor abc to a, b, c to accommodate individual feature ranges by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/43
* Grayskull conda forge by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/48
* replace USERNAME with sparks-baird and `conda-forge` instructions by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/47


## New Contributors
* @hasan-sayeed made their first contribution in https://github.com/sparks-baird/xtal2png/pull/39

**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.2.1...v0.3.0

### Bug Fixes

- general:
  - fixup colab badge link ([88727ed](https://github.com/sparks-baird/xtal2png/commit/88727ede42b70014be85e70338f58e3be4399d95))

### Refactor

- general:
  - refactor abc to a, b, c to accommodate individual feature ranges ([9bec071](https://github.com/sparks-baird/xtal2png/commit/9bec0710580d487c49c71e26f5d2f257b0b9e5a0)) ([#43](https://github.com/sparks-baird/xtal2png/pull/43))

## [v0.2.1](https://github.com/sparks-baird/xtal2png/releases/tag/v0.2.1) - 2022-05-28 17:12:26


<!-- Release notes generated using configuration in .github/release.yml at v0.2.1 -->



**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.2.0...v0.2.1

## [v0.2.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.2.0) - 2022-05-28 09:21:45

- no changes

<!-- Release notes generated using configuration in .github/release.yml at v0.2.0 -->

## What's Changed
### Other Changes
* sphinx_rtd_theme, icon / favicon, and other conf.py settings by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/31
* forgot f in f-string in ValueError by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/32
* self.max_sites instead of max_sites by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/33
* use a larger logo by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/34
* use .ico and fixup path by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/35
* working cli via `xtal2png --encode ...` and `xtal2png --decode ...` syntax by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/37


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.6...v0.2.0

## [v0.1.6](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.6) - 2022-05-27 05:36:46

- no changes

<!-- Release notes generated using configuration in .github/release.yml at v0.1.6 -->

## What's Changed
### Other Changes
* minor typos, clarification by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/29
* Fix png2xtal shape mismatch by np.stack-ing on correct axis (and add unit tests) by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/30


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.5...v0.1.6

### Bug Fixes

- general:
  - fix logic for when to np.squeeze arrays after disassembly ([ae48ce3](https://github.com/sparks-baird/xtal2png/commit/ae48ce3b32bf30bb27b8338c9f29c4a381269eeb)) ([#30](https://github.com/sparks-baird/xtal2png/pull/30))

## [v0.1.5](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.5) - 2022-05-27 04:39:02

<!-- Release notes generated using configuration in .github/release.yml at v0.1.5 -->

## What's Changed
### Other Changes
* resolve save_dir not found by making directory if it doesn't exist by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/28


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.4...v0.1.5

## [v0.1.4](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.4) - 2022-05-27 04:04:54

<!-- Release notes generated using configuration in .github/release.yml at v0.1.4 -->

## What's Changed
### Other Changes
* Conda forge workflow - use grayskull to create `meta.yaml` by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/23
* Conda forge workflow, first working pass by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/24
* Create codeql-analysis.yml by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/25
* Create release.yml by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/26
* refactor to latest extension version, uses myst-parser instead by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/27


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.3...v0.1.4

### Refactor

- general:
  - refactor to latest extension version, uses myst-parser instead ([5d31015](https://github.com/sparks-baird/xtal2png/commit/5d31015313115a8573f11c0b05b5cbf66000e77f)) ([#27](https://github.com/sparks-baird/xtal2png/pull/27))

## [v0.1.3](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.3) - 2022-05-24 05:08:44

`setup.cfg` updated
**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.2...v0.1.3

## [v0.1.2](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.2) - 2022-05-24 04:57:00

## What's Changed
* initial release and console script fixups by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/19


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.1...v0.1.2

### Bug Fixes

- general:
  - fixup cli usage (untested) ([beb67bc](https://github.com/sparks-baird/xtal2png/commit/beb67bc277b1c27ae5248867aa64c814b34d1b5f)) ([#22](https://github.com/sparks-baird/xtal2png/pull/22))
  - fixup console script fn ([ce23252](https://github.com/sparks-baird/xtal2png/commit/ce232520ec4eeffc5e860c20480249f603c8dd8b)) ([#19](https://github.com/sparks-baird/xtal2png/pull/19))
  - fixup accidental hardcoded "2" and replace with `n_structures` ([dd4cccd](https://github.com/sparks-baird/xtal2png/commit/dd4cccdd51b825f2e5532cfe4726aaf96ef11181)) ([#15](https://github.com/sparks-baird/xtal2png/pull/15))
  - fixup atom range, move commented code to graveyard, ([b006510](https://github.com/sparks-baird/xtal2png/commit/b0065100e556cbc61a53844d048412af535a187a)) ([#3](https://github.com/sparks-baird/xtal2png/pull/3))
  - fixing up fit_transform ([ff6a90d](https://github.com/sparks-baird/xtal2png/commit/ff6a90d1187fdd31c203d1001d18e02f28b32bf7)) ([#3](https://github.com/sparks-baird/xtal2png/pull/3))

### Documentation

- general:
  - docstring example ([b8fb3d5](https://github.com/sparks-baird/xtal2png/commit/b8fb3d57723b4f42a915354d42e2c89f24f8aa33)) ([#6](https://github.com/sparks-baird/xtal2png/pull/6))
  - docstrings, __init__ method, xtal2png ([d1213f1](https://github.com/sparks-baird/xtal2png/commit/d1213f147c9461961494b601516c2eead3edbaac)) ([#6](https://github.com/sparks-baird/xtal2png/pull/6))
  - docstrings too ([23056b4](https://github.com/sparks-baird/xtal2png/commit/23056b48eea1aab284a3ddfc3a80c739f88f371f)) ([#6](https://github.com/sparks-baird/xtal2png/pull/6))

\* *This CHANGELOG was automatically generated by [auto-generate-changelog](https://github.com/BobAnkh/auto-generate-changelog)*
