# CHANGELOG

## Unreleased

Changes unreleased.

### Bug Fixes

- general:
  - fixup joss badge, remove leftover \autoref command ([0b4667a](https://github.com/sparks-baird/xtal2png/commit/0b4667a01da9944bc9786560a4098d8df5985c84)) ([#172](https://github.com/sparks-baird/xtal2png/pull/172))

## [v0.8.0](https://github.com/sparks-baird/xtal2png/releases/tag/untagged-1cfe4c6287b8c846dc48) - 2022-07-08 07:13:50

<!-- Release notes generated using configuration in .github/release.yml at v0.8.0 -->

## What's Changed

### Features
* [WIP] feat: encode and decode using `element_coder` by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/117
* replace volume with num_sites and use num_sites as a mask by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/141
* allow for masking out redundant info from representation by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/138
* public api / top-level imports since not using namespace packages by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/143

### JOSS Paper
* Add Kevin Jablonka @kjappelbaum and Colton Seegmiller @cseeg as co-authors to JOSS manuscript by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/148
* Response to PeterKraus software paper review comments by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/147

### Notebooks/Documentation
* Update 2.1-xtal2png-cnn-classification.ipynb by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/158
* jupyter notebooks and python scripts surfaced in docs via nbsphinx by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/151

### Other Changes
* png2xtal: if not isinstance(images, list) raise ValueError by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/137
* max_sites error msg: note about encode/decode_cell_type kwargs by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/139
* test_max_sites by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/140
* mention XtalConverter as top-level API, and other links to API by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/150
* move toctree to examples.md and use nbsphinx recommended format by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/152
* docs: paper edits by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/153
* fixup toctree syntax by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/155
* nbsphinx_link and scripts in examples tab by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/156
* add ipykernel as docs req by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/157
* fixup files and content for nbsphinx_link by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/159
* add title to imagen-pytorch nb by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/160
* touching up presentation of notebooks within docs by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/161
* fix typo (as --> and) and syntax error in 2 references for smiles/selfies by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/163
* split xtal2png_test into multiple modules and test loading saved images by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/164
* Create parameters-notes.txt by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/166
* remove leftover sys.stdout restoration by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/165
* feat: implement click CLI by @kjappelbaum in https://github.com/sparks-baird/xtal2png/pull/154
* rm code graveyard in core.py by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/169
* maxdepth should be 2 for the examples toctree by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/170

## New Contributors
* @kjappelbaum made their first contribution in https://github.com/sparks-baird/xtal2png/pull/117

**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.7.0...v0.8.0

### Feature

- general:
  - implement click CLI ([62511f7](https://github.com/sparks-baird/xtal2png/commit/62511f7eeda47771a39b2332303f8b580a75107e)) ([#154](https://github.com/sparks-baird/xtal2png/pull/154))

### Bug Fixes

- general:
  - fixup test, loading saved images ([4ce4fe5](https://github.com/sparks-baird/xtal2png/commit/4ce4fe58c89938e556c5a895093774b8711a6cb9)) ([#164](https://github.com/sparks-baird/xtal2png/pull/164))
  - fix typo (as --> and) and syntax error in 2 references for smiles/selfies ([390f92e](https://github.com/sparks-baird/xtal2png/commit/390f92eae43abe75b84ff5816fdcb9ff4bf24529)) ([#163](https://github.com/sparks-baird/xtal2png/pull/163))
  - fix typo in path ([403ae79](https://github.com/sparks-baird/xtal2png/commit/403ae79060367bc0a908ade433fe84c59a2bace5)) ([#161](https://github.com/sparks-baird/xtal2png/pull/161))
  - fixup files and content for nbsphinx_link ([4d619b6](https://github.com/sparks-baird/xtal2png/commit/4d619b6fc7a351545d38e650459d16da4e90c11d)) ([#159](https://github.com/sparks-baird/xtal2png/pull/159))
  - fix eof ([0529235](https://github.com/sparks-baird/xtal2png/commit/0529235f7946ccbc857486880c5f8f398de77789)) ([#157](https://github.com/sparks-baird/xtal2png/pull/157))
  - fixup syntax ([f29d0ac](https://github.com/sparks-baird/xtal2png/commit/f29d0ac68b7058d210dcb17fdd12cce2bd06fed9)) ([#155](https://github.com/sparks-baird/xtal2png/pull/155))
  - also allow single file ([55f3a94](https://github.com/sparks-baird/xtal2png/commit/55f3a94ba5e8ff90121c300d31a5bb3ad5b91472)) ([#154](https://github.com/sparks-baird/xtal2png/pull/154))
  - click.Path options ([f795e54](https://github.com/sparks-baird/xtal2png/commit/f795e54ced48c9a397d2543a4bb378fa4f707d0f)) ([#154](https://github.com/sparks-baird/xtal2png/pull/154))
  - fixup bug with affiliation and add Colton's orcid ([ab78ab7](https://github.com/sparks-baird/xtal2png/commit/ab78ab756344355911d4e8ec53ee224f85d8de05)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - fixup :func: syntax ([02af1d0](https://github.com/sparks-baird/xtal2png/commit/02af1d039d52994a842a0e878a56c1f094ed40d4)) ([#150](https://github.com/sparks-baird/xtal2png/pull/150))
  - fixup num_sites fitting ([a6579fd](https://github.com/sparks-baird/xtal2png/commit/a6579fde90b420d4db1f8281364edad474cc49f6)) ([#141](https://github.com/sparks-baird/xtal2png/pull/141))
  - fixup getting started instructions, linting, mention m3gnet as optional ([08ac35a](https://github.com/sparks-baird/xtal2png/commit/08ac35ac42d448a359ae18eed7a6d400074cadee)) ([#143](https://github.com/sparks-baird/xtal2png/pull/143))
  - imports ([8a7a51b](https://github.com/sparks-baird/xtal2png/commit/8a7a51bf338005637b2f90a9c60df906ae8ca8d5)) ([#117](https://github.com/sparks-baird/xtal2png/pull/117))

### Documentation

- general:
  - update readme cli example ([fe955b2](https://github.com/sparks-baird/xtal2png/commit/fe955b28816fd3b5ed74b92ec1bfe0978c1cfd15)) ([#154](https://github.com/sparks-baird/xtal2png/pull/154))
  - dating in -> dating to (according to Grammarly) ([1b78ade](https://github.com/sparks-baird/xtal2png/commit/1b78adef13fdf2f715d82222668c1859f1edbf51)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - crystal structure -> crystal structures ([cdbc812](https://github.com/sparks-baird/xtal2png/commit/cdbc8123720c52bafa38efe1f3c5cb28b00e717c)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - add citation to SELFIES review/perspective ([dc2b4e3](https://github.com/sparks-baird/xtal2png/commit/dc2b4e31746510b01e5417d35850b7fb88f91396)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - crystal structure -> crystal structures ([7a89f43](https://github.com/sparks-baird/xtal2png/commit/7a89f43f29b0590f53191cb5f160cb235f8af344)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - correlaries does not exist in Merriam-Webster dictionary ([de15339](https://github.com/sparks-baird/xtal2png/commit/de153391bb11fc5e36b61a0b0025d44bccdddfa7)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - grayscale in parenthesis as it might also be RGB (https://github.com/sparks-baird/xtal2png/pull/111) ([f18e639](https://github.com/sparks-baird/xtal2png/commit/f18e639fb72754946bee64f656c7d936f1289ce3)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - only mention first date for clarity ([8f61d5e](https://github.com/sparks-baird/xtal2png/commit/8f61d5e65fdcd4850ea59fc8216e839a108a770e)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))
  - run Antidote spell check ([d0deb1f](https://github.com/sparks-baird/xtal2png/commit/d0deb1f4e24fa3eec7ce6b0241cba5547e4694f0)) ([#153](https://github.com/sparks-baird/xtal2png/pull/153))

## [v0.7.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.7.0) - 2022-06-23 04:40:25

<!-- Release notes generated using configuration in .github/release.yml at v0.7.0 -->

## What's Changed
* convert encode/decode_as_primitive kwargs to encode/decode_cell_type kwargs by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/131

**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.6.3...v0.7.0

## [v0.6.3](https://github.com/sparks-baird/xtal2png/releases/tag/v0.6.3) - 2022-06-23 04:09:02

<!-- Release notes generated using configuration in .github/release.yml at v0.6.3 -->

## What's Changed
* encode/decode as primitive True by default (temporary fix) by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/130
* update imagen-pytorch example by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/123

### Other Changes
* Update CONTRIBUTING.md by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/121
* remove todo's from CONTRIBUTING.md by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/124
* add badges and other image-to-image models to index.md by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/125
* fixup badges in index.md by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/126
* materials project feature ranges based on conventional unit cells by @hasan-sayeed in https://github.com/sparks-baird/xtal2png/pull/114
* replace erroneous "made with pymatviz" with ase in paper.md by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/127


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.6.2...v0.6.3

### Bug Fixes

- general:
  - fixup badges in index.md ([1fcf537](https://github.com/sparks-baird/xtal2png/commit/1fcf537c1b52474a8728e28d91b40bf8eafd76d3)) ([#126](https://github.com/sparks-baird/xtal2png/pull/126))

### Refactor

- general:
  - refactor to unit_cell_converter function ([d7963fa](https://github.com/sparks-baird/xtal2png/commit/d7963fabf511031949e541f8095573fb878bf8c2)) ([#131](https://github.com/sparks-baird/xtal2png/pull/131))
  - refactor multiple for loops into single for loop and enhance mem efficiency ([1653229](https://github.com/sparks-baird/xtal2png/commit/1653229e7fc187c8d10f81cdd8e71a6dd54bc568)) ([#130](https://github.com/sparks-baird/xtal2png/pull/130))

## [v0.6.2](https://github.com/sparks-baird/xtal2png/releases/tag/v0.6.2) - 2022-06-20 18:47:21

<!-- Release notes generated using configuration in .github/release.yml at v0.6.2 -->

## What's Changed
### Other Changes
* try except for m3gnet import if relax_on_decode by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/110
* refactor to use rgb_scaling=False in imagen-pytorch example by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/107
* support RGB (3-channel) averaging in addition to grayscale images by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/111
* add num_sites and a note about displaying plotly figures by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/115
* flat is better than nested (fit refactor) by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/113
* commented "shuffle" code (might test later) and add channels variable to example scripts by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/112
* make m3gnet an optional dep and relax_on_decode False by default by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/118
* tqdm_if_verbose wrapper fn by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/120


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.5.1...v0.6.2

### Feature

- general:
  - encode and decode using ([fc39868](https://github.com/sparks-baird/xtal2png/commit/fc3986868f466c392e7ca7129731249c91c9ea53)) ([#117](https://github.com/sparks-baird/xtal2png/pull/117))

### Refactor

- general:
  - refactor to use rgb_scaling=False ([04ba552](https://github.com/sparks-baird/xtal2png/commit/04ba5525646dc2c7afa5e22e90855ea9998b9a4d)) ([#107](https://github.com/sparks-baird/xtal2png/pull/107))

## [v0.5.1](https://github.com/sparks-baird/xtal2png/releases/tag/v0.5.1) - 2022-06-18 02:40:01

<!-- Release notes generated using configuration in .github/release.yml at v0.5.1 -->

## What's Changed
### Other Changes
* structures <---> arrays functions get rgb_output kwarg, default True by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/104
* imagen-pytorch example by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/106
* Add tests for rgb_scaling=False by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/105


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.5.0...v0.5.1

## [v0.5.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.5.0) - 2022-06-17 04:13:38

<!-- Release notes generated using configuration in .github/release.yml at v0.5.0 -->

## What's Changed
* ❗Implement [`m3gnet`](https://github.com/materialsvirtuallab/m3gnet)'s [DFT surrogate structure relaxation](https://github.com/materialsvirtuallab/m3gnet#structure-relaxation) during decoding and use by default by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/100.
  * (Much thanks to @shyuep for [support on this](https://github.com/materialsvirtuallab/m3gnet/issues?q=is%3Aissue+commenter%3Asgbaird))
* fit method for feature ranges by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/91 as suggested by @kjappelbaum
* encode vs. decode symprec, angle_tolerance, and primitive options by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/97
* Use pymatgen StructureMatcher as initial check and preprocesser before detailed matching by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/85
* Use get_refined_structure with symprec and angle_tolerance during decoding by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/86
* denoising diffusion notebook and script by @sgbaird and @hasan-sayeed in https://github.com/sparks-baird/xtal2png/pull/70

### Other Changes
* Paper by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/67
* remove code graveyards by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/68
* minor joss paper fixes by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/69
* update paper affiliations by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/72
* Update run_grayskull.py with section order sorting by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/71
* replace strange characters with normal characters in chemical formula by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/74
* denoising diffusion probabilistic model script by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/73
* Create 2.2-xgboost-matbench-benchmark.ipynb by @sgbaird and @cseeg in https://github.com/sparks-baird/xtal2png/pull/78
* running ddpm from pretrained model and sampling by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/77
* plot of the equimolar elemental contributions using pymatviz by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/87
* fixup :func: reference by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/88
* Update 2.0-materials-project-feature-ranges.ipynb with output cells by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/90
* add attributions section by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/92
* links to google imagen by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/93
* fill distance_matrix diagonals with zeros in arrays_to_structures by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/95
* modify `structures` directly instead of appending to a new list by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/99
* update README and ipynb with m3gnet usage (/fixup syntax) by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/101

**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.4.0...v0.5.0

### Bug Fixes

- general:
  - fixup: loop through stacked distance matrices ([f88c0ff](https://github.com/sparks-baird/xtal2png/commit/f88c0ff83caf9f447a613caf5e035c7d748fafd7)) ([#95](https://github.com/sparks-baird/xtal2png/pull/95))
  - fix mistake where low_val should have been upp_val ([323fc5d](https://github.com/sparks-baird/xtal2png/commit/323fc5de0498b50015ab61b64060d0240eb39d83)) ([#91](https://github.com/sparks-baird/xtal2png/pull/91))
  - fixup :func: reference ([cdd3b88](https://github.com/sparks-baird/xtal2png/commit/cdd3b886d241e165346f013eb0bb62d85434325a)) ([#88](https://github.com/sparks-baird/xtal2png/pull/88))
  - fix joss submitted badge ([72f44ef](https://github.com/sparks-baird/xtal2png/commit/72f44ef91c1668a9be4988fdb0f5b0afd8ad134e))
  - fix typos, remove extraneous figure, add future work ([d0ffb91](https://github.com/sparks-baird/xtal2png/commit/d0ffb917e062c0a580a42dc5ee5309c4e036e48f)) ([#67](https://github.com/sparks-baird/xtal2png/pull/67))

## [v0.4.0](https://github.com/sparks-baird/xtal2png/releases/tag/v0.4.0) - 2022-06-03 02:01:29

<!-- Release notes generated using configuration in .github/release.yml at v0.4.0 -->

## What's Changed
### Other Changes
* available on conda-forge by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/49
* fix colab link in index.md by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/53
* add colab link to README by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/52
* fixup colab link by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/56
* add note about needing development versions of pyscaffold and extensions by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/58
* Support rgb images in `png2xtal` by converting to grayscale by @sgbaird in https://github.com/sparks-baird/xtal2png/pull/62


**Full Changelog**: https://github.com/sparks-baird/xtal2png/compare/v0.3.0...v0.4.0

### Bug Fixes

- general:
  - fixup colab link ([7803383](https://github.com/sparks-baird/xtal2png/commit/7803383b2e1acbb565b417d82606a6c40ac425b8)) ([#56](https://github.com/sparks-baird/xtal2png/pull/56))
  - fix colab link in index.md ([7ff9bfa](https://github.com/sparks-baird/xtal2png/commit/7ff9bface93181f40fcf9e733dd082f737249b74)) ([#53](https://github.com/sparks-baird/xtal2png/pull/53))

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
