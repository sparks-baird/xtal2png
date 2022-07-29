# CHANGELOG

## [v0.9.1](https://github.com/sparks-baird/xtal2png/releases/tag/v0.9.1) - 2022-07-28 22:06:22

<!-- Release notes generated using configuration in .github/release.yml at v0.9.1 -->

## What's Changed
* cnn-classification notebook (Created using Colaboratory) by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/175
* add regression notebook and change classification Colab badge to markdown for compatibility with nbsphinx by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/178
* fix: catch partial occupancy by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/182
* fix: spacegroup fallback and installation docs  by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/186
* feat: add `max_sites` to CLI by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/181

### Other Changes
* fixup joss badge, remove leftover \autoref command by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/172
* split large block of README text into subsections by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/171
* convert colab badges from html to markdown for nbsphinx compatibility by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/173
* Revert "cnn-classification notebook (Created using Colaboratory)" by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/176
* Created using Colaboratory by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/177
* paper: update software paper by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/183
* JOSS paper review - Docs by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/187
* add Berend as co-author by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/191
* Revert imports until conda-forge is bumped by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/189
* response to rkurchin's editorial review by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/192


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.8.0...v0.9.1

### Feature

- general:
  - add `max_sites` to CLI ([27c39ec](https://github.com/sparks-baird/xtal2png/commit/27c39ec7310525cfe8205cadcbdd2add7e8e964f)) ([#181](https://github.com/sparks-baird/xtal2png/pull/181))

### Bug Fixes

- general:
  - if space group cannot be found, return 0 ([dbb2e68](https://github.com/sparks-baird/xtal2png/commit/dbb2e68b7a380d9ef03b2ce3b1fb9452d3a14f90)) ([#186](https://github.com/sparks-baird/xtal2png/pull/186))
  - wrong filepath in text ([5e151ac](https://github.com/sparks-baird/xtal2png/commit/5e151ac03767ab157eb927ff767957bd9d28aae1)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - broken instance check ([f02d4b4](https://github.com/sparks-baird/xtal2png/commit/f02d4b433285a7f503770f688a5aa5bef47dac55)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - broken instance check ([8067094](https://github.com/sparks-baird/xtal2png/commit/80670949bc5162b74b53b2682f2f09690cb05dec)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - add test file of disordered structure ([6af92ba](https://github.com/sparks-baird/xtal2png/commit/6af92ba97188d0558cd2b4a5012fac44052c82c9)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - add test file of disordered structure ([b0b4374](https://github.com/sparks-baird/xtal2png/commit/b0b437494fe095ddf4f215ca02561cf952368c0e)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - add fixture decorator ([9e6bcad](https://github.com/sparks-baird/xtal2png/commit/9e6bcad7d0cdba37baa7a80134a6ed3c7bf31b1d)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - catch partial occupancy ([449f949](https://github.com/sparks-baird/xtal2png/commit/449f9498265fb98ae4664d0c844bb4e0ff82cef7)) ([#182](https://github.com/sparks-baird/xtal2png/pull/182))
  - fixup joss badge, remove leftover \autoref command ([0b4667a](https://github.com/sparks-baird/xtal2png/commit/0b4667a01da9944bc9786560a4098d8df5985c84)) ([#172](https://github.com/sparks-baird/xtal2png/pull/172))

### Documentation

- general:
  - comment out pending conda installation ([0641b7c](https://github.com/sparks-baird/xtal2png/commit/0641b7c1230e7b7e1f00840c227d0c4134d610fa)) ([#186](https://github.com/sparks-baird/xtal2png/pull/186))
  - line 33 ([c5c49f6](https://github.com/sparks-baird/xtal2png/commit/c5c49f67327b8a6a65576012915e0d5aeb556f49)) ([#183](https://github.com/sparks-baird/xtal2png/pull/183))
  - line 30 ([c89a900](https://github.com/sparks-baird/xtal2png/commit/c89a900c3ff90ca397f69552c5cdd69a4993c240)) ([#183](https://github.com/sparks-baird/xtal2png/pull/183))
  - Jun -> June ([dfb1f36](https://github.com/sparks-baird/xtal2png/commit/dfb1f36ad6af956998ac8ec7d84a6759567aace7)) ([#183](https://github.com/sparks-baird/xtal2png/pull/183))

## [v0.8.0](https://github.com/sparks-baird/xtal2png/releases/tag/untagged-1cfe4c6287b8c846dc48) - 2022-07-08 07:13:50

<!-- Release notes generated using configuration in .github/release.yml at v0.8.0 -->

## [v0.7.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.7.0) - 2022-06-23 04:40:25

<!-- Release notes generated using configuration in .github/release.yml at v0.7.0 -->

## [v0.6.3](https://github.com/sparks-baird/xtal2png/releases/tag/v0.6.3) - 2022-06-23 04:09:02

<!-- Release notes generated using configuration in .github/release.yml at v0.6.3 -->

## [v0.6.2](https://github.com/sparks-baird/xtal2png/releases/tag/v0.6.2) - 2022-06-20 18:47:21

<!-- Release notes generated using configuration in .github/release.yml at v0.6.2 -->

## [v0.5.1](https://github.com/sparks-baird/xtal2png/releases/tag/v0.5.1) - 2022-06-18 02:40:01

<!-- Release notes generated using configuration in .github/release.yml at v0.5.1 -->

## [v0.5.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.5.0) - 2022-06-17 04:13:38

<!-- Release notes generated using configuration in .github/release.yml at v0.5.0 -->

## [v0.4.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.4.0) - 2022-06-03 02:01:29

<!-- Release notes generated using configuration in .github/release.yml at v0.4.0 -->

## [v0.3.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.3.0) - 2022-05-31 17:28:49

<!-- Release notes generated using configuration in .github/release.yml at v0.3.0  -->

## [v0.2.1](https://github.com/sparks-baird/xtal2png/releases/tag/v0.2.1) - 2022-05-28 17:12:26


<!-- Release notes generated using configuration in .github/release.yml at v0.2.1 -->



**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.2.0...v0.2.1

## [v0.2.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.2.0) - 2022-05-28 09:21:45

- no changes

<!-- Release notes generated using configuration in .github/release.yml at v0.2.0 -->

## [v0.1.6](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.6) - 2022-05-27 05:36:46

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

## [v0.1.4](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.4) - 2022-05-27 04:04:54

<!-- Release notes generated using configuration in .github/release.yml at v0.1.4 -->

## [v0.1.3](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.3) - 2022-05-24 05:08:44

`setup.cfg` updated
**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.1.2...v0.1.3

## [v0.1.2](https://github.com/sparks-baird/xtal2png/releases/tag/v0.1.2) - 2022-05-24 04:57:00

\* *This CHANGELOG was automatically generated by [auto-generate-changelog](https://github.com/BobAnkh/auto-generate-changelog)*
