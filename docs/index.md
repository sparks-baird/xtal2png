# xtal2png

Encode/decode a crystal structure to/from a grayscale PNG image for direct use with image-based machine learning models such as [Imagen], [DALLE2], or [Palette].[^1]

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sparks-baird/xtal2png/blob/main/notebooks/1.0-xtal2png-tutorial.ipynb)
[![status](https://joss.theoj.org/papers/0c704f6ae9739c1e97e05ae0ad57aecb/status.svg)](https://joss.theoj.org/papers/0c704f6ae9739c1e97e05ae0ad57aecb)
[![PyPI -
Downloads](https://img.shields.io/pypi/dm/xtal2png)](https://pypi.org/project/xtal2png)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/xtal2png?style=flat&color=blue&label=Conda%20Downloads)](https://anaconda.org/conda-forge/xtal2png)

<a class="github-button" href="https://github.com/sparks-baird/xtal2png"
data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star
sparks-baird/xtal2png on GitHub">Star</a>
<a class="github-button"
href="https://github.com/sgbaird" data-size="large" data-show-count="true"
aria-label="Follow @sgbaird on GitHub">Follow @sgbaird</a>
<a class="github-button" href="https://github.com/sparks-baird/xtal2png/issues"
data-icon="octicon-issue-opened" data-size="large" data-show-count="true"
aria-label="Issue sparks-baird/xtal2png on GitHub">Issue</a>
<a class="github-button" href="https://github.com/sparks-baird/xtal2png/discussions" data-icon="octicon-comment-discussion" data-size="large" aria-label="Discuss sparks-baird/xtal2png on GitHub">Discuss</a>
<br><br>

The latest advances in machine learning are often in natural language such as with long
short-term memory networks (LSTMs) and transformers or image processing such as with
generative adversarial networks (GANs), variational autoencoders (VAEs), and guided
diffusion models; however, transfering these advances to adjacent domains such as
materials informatics often takes years. `xtal2png` encodes and decodes crystal
structures via grayscale PNG images by writing and reading the necessary information for
crystal reconstruction (unit cell, atomic elements, atomic coordinates) as a square
matrix of numbers, respectively. This is akin to making/reading a QR code for crystal
structures, where the `xtal2png` representation is invertible. The ability to feed these
images directly into image-based pipelines allows you, as a materials informatics
practitioner, to get streamlined results for new state-of-the-art image-based machine
learning models applied to crystal structure.

> Results manuscript coming soon!

 <!-- ![GitHub Repo stars](https://img.shields.io/github/stars/sparks-baird/xtal2png?style=social) ![GitHub followers](https://img.shields.io/github/followers/sgbaird?style=social) ![GitHub issues](https://img.shields.io/github/issues-raw/sparks-baird/xtal2png) ![GitHub closed issues](https://img.shields.io/github/issues-closed-raw/sparks-baird/xtal2png) -->

## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Examples <examples>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[MyST]: https://myst-parser.readthedocs.io/en/latest/
[Palette]: https://iterative-refinement.github.io/palette/
[Janspiry/Palette-Image-to-Image-Diffusion-Models]: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models
[Imagen]: https://imagen.research.google/
[lucidrains/imagen-pytorch]: https://github.com/lucidrains/imagen-pytorch#usage
[DALLE2]: https://openai.com/dall-e-2/
[lucidrains/DALLE2-pytorch]: https://github.com/lucidrains/DALLE2-pytorch#unconditional-training
[^1]: For unofficial implementations, see [lucidrains/imagen-pytorch], [lucidrains/DALLE2-pytorch], and [Janspiry/Palette-Image-to-Image-Diffusion-Models], respectively

<script async defer src="https://buttons.github.io/buttons.js"></script>
