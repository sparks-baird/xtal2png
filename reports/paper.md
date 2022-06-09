---
title: 'xtal2png: A Python package for representing crystal structure as PNG files'
tags:
  - Python
  - materials informatics
  - crystal structure
  - computer vision
  - image-based predictions
authors:
  - name: Sterling G. Baird
    orcid: 0000-0002-4491-6876
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Hasan M. Sayeed
    orcid: 0000-0002-6583-7755
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Michael Alverson
    orcid: 0000-0002-4857-7584
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Taylor D. Sparks
    orcid: 0000-0001-8020-7711
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Materials Science & Engineering, University of Utah, Salt Lake City, USA
   index: 1
date: 6 July 2022
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The latest advances in machine learning are often in natural language such as with long
short-term memory networks (LSTMs) and transformers or image processing such as with
generative adversarial networks (GANs), variational autoencoders (VAEs), and guided
diffusion models. Encoding/decoding crystal structures via grayscale PNG images (see
e.g. \autoref{fig:64-bit}) is akin to making/reading a QR code for crystal structures.
This allows you, as a materials informatics practitioner, to get streamlined results for
new state-of-the-art image-based machine learning models applied to crystal structure.

![A real-size $64\times64$ pixel `xtal2png` representation of a crystal structure.\label{fig:64-bit}](figures/Zn8B8Pb4O24,volume=623,uid=bc2d.png)

<!-- The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration). -->

# Statement of need

Using a state-of-the-art method in a separate domain with a custom data representation
is often an expensive and drawn-out process. For example, @vaswaniAttentionAllYou2017
introduced the revolutionary natural language processing transformer architecture in Jun
2017; yet the application of transformers to the adjacent domain of materials
informatics (chemical-formula-based predictions) was not publicly realized until late
2019/early 2020
[@goodallPredictingMaterialsProperties2019,@wangCompositionallyrestrictedAttentionbasedNetwork2020],
approximately two-and-a-half years later, with peer-reviewed publications dating in late
2020/ mid-2021
[@goodallPredictingMaterialsProperties2020,@wangCompositionallyRestrictedAttentionbased2021].
Another example of state-of-the-art algorithm domain transfer is refactoring
state-of-the-art image-processing models for crystal structure applications, with
introduction [@kipfSemisupervisedClassificationGraph2016], domain transfer (preprint)
[@xieCrystalGraphConvolutional2017], and peer-reviewed domain transferred
[@xieCrystalGraphConvolutional2018] publication dates of Sep 2016, Oct 2017, and Apr
2018, respectively. Here, we focus on the latter application: state-of-the-art domain
transfer from image-processing to crystal structure.

`xtal2png` is a Python package that allows you to encode/decode a crystal structure
to/from a grayscale PNG image for direct use with image-based machine learning models.
For example, Let's take [Google's image-to-image diffusion model,
Palette](https://iterative-refinement.github.io/palette/)
[@sahariaPaletteImagetoImageDiffusion2022]. Rather than dig into the code spending
hours, days, or weeks modifying, debugging, and playing GitHub phone tag with the
developers before you can (maybe) get preliminary results, `xtal2png` lets you get those
results using the default instructions on the repository, assuming the instructions can
be run without error.

![(a) upscaled example image and (b) legend of the `xtal2png` encoding.\label{fig:example-and-legend}](figures/example-and-legend.png)

`xtal2png` was designed to be easy-to-use by both
["Pythonistas"](https://en.wiktionary.org/wiki/Pythonista) and entry-level coders alike.
`xtal2png` provides a straightforward Python application programming interface (API) and
command line interface (CLI). `xtal2png` relies on `pymatgen.core.structure.Structure`
objects for representing crystal structures and also supports reading crystallographic
information files (CIFs) from directories. `xtal2png` encodes crystallographic
information related to the unit cell, crystallographic symmetry, and atomic elements and
coordinates which are each scaled individually according to the information type. An
upscaled version of the PNG image and a legend of the representation are given in
\autoref{fig:example-and-legend}. Due to the encoding of numerical values as grayscale
PNG images (allowable values are integers between 0 and 255), a small round-off error is
present during a single round of encoding and decoding. Original and decoded
visualizations of the crystal structure represented in \autoref{fig:example-and-legend}
are given in \autoref{fig:original-decoded}. The significance of the representation lies
in being able to directly use the PNG representation with image-based models which often
do not directly support custom dataset types, potentially saving days or weeks during
the process of obtaining preliminary results on a newly released model.

<!-- | (a) 64x64 pixels | (b) Scaled for Better Viewing ([tool credit](https://lospec.com/pixel-art-scaler/)) |  Legend |
| --- | --- | --- |
| ![Zn8B8Pb4O24,volume=623,uid=bc2d](figures/Zn8B8Pb4O24,volume=623,uid=bc2d.png) | <img src="figures/Zn8B8Pb4O24,volume=623,uid=bc2d_upsampled.png" width=400> | <img src="figures/legend.png" width=400> | -->

<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->


<!-- `Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike. -->

<!-- # Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

<!--
# Citations
Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

<!-- # Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements

This work was supported by the National Science Foundation, USA under Grant No. DMR-1651668.

# References