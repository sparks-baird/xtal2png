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
  - name: Michael D. Alverson
    orcid: 0000-0002-4857-7584
    equal-contrib: false
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Taylor D. Sparks
    orcid: 0000-0001-8020-7711
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Materials Science & Engineering, University of Utah, USA
   index: 1
 - name: Computer Science, University of Southern California, USA
   index: 2
date: 9 June 2022
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
diffusion models. Using `xtal2png` [@xtal2png] to encode/decode crystal structures via grayscale PNG images (see
e.g. \autoref{fig:64-bit}) is akin to making/reading a QR code for crystal structures.
This allows you, as a materials informatics practitioner, to get streamlined results for
new state-of-the-art image-based machine learning models applied to crystal structure.

![A real-size $64\times64$ pixel `xtal2png` representation of a crystal structure.\label{fig:64-bit}](figures/Zn8B8Pb4O24,volume=623,uid=bc2d.png)

# Statement of need

Using a state-of-the-art method in a separate domain with a custom data representation
is often an expensive and drawn-out process. For example, @vaswaniAttentionAllYou2017
introduced the revolutionary natural language processing transformer architecture in Jun
2017; yet the application of transformers to the adjacent domain of materials
informatics (chemical-formula-based predictions) was not publicly realized until late
2019/early 2020
[@goodallPredictingMaterialsProperties2019;@wangCompositionallyrestrictedAttentionbasedNetwork2020],
approximately two-and-a-half years later, with peer-reviewed publications dating in late
2020/ mid-2021
[@goodallPredictingMaterialsProperties2020;@wangCompositionallyRestrictedAttentionbased2021].
Another example of state-of-the-art algorithm domain transfer is refactoring image-processing models for crystal structure applications, with
introduction [@kipfSemisupervisedClassificationGraph2016], domain transfer (preprint)
[@xieCrystalGraphConvolutional2017], and peer-reviewed domain transferred
[@xieCrystalGraphConvolutional2018] publication dates of Sep 2016, Oct 2017, and Apr
2018, respectively. Here, we focus on the latter application: state-of-the-art domain
transfer from image-processing to crystal structure.

`xtal2png` [@xtal2png]
([https://github.com/sparks-baird/xtal2png](https://github.com/sparks-baird/xtal2png))
is a Python package that allows you to encode/decode a crystal structure to/from a
grayscale PNG image for direct use with image-based machine learning models. Let's take
[Google's image-to-image diffusion model,
Palette](https://iterative-refinement.github.io/palette/)
[@sahariaPaletteImagetoImageDiffusion2022]. Rather than dig into the code spending
hours, days, or weeks modifying, debugging, and playing GitHub phone tag with the
developers before you can (maybe) get preliminary results, `xtal2png` lets you get those
results using the default instructions on the repository, assuming the instructions can
be run without error.

![(a) upscaled example image and (b) legend of the `xtal2png` encoding.\label{fig:example-and-legend}](figures/example-and-legend.png)

`xtal2png` was designed to be easy-to-use by both
"[Pythonistas](https://en.wiktionary.org/wiki/Pythonista)" and entry-level coders alike.
`xtal2png` provides a straightforward Python application programming interface (API) and
command line interface (CLI). `xtal2png` relies on `pymatgen.core.structure.Structure`
[@ongPythonMaterialsGenomics2013] objects for representing crystal structures and also
supports reading crystallographic information files (CIFs) from directories. `xtal2png`
encodes crystallographic information related to the unit cell, crystallographic
symmetry, and atomic elements and coordinates which are each scaled individually
according to the information type. An upscaled version of the PNG image and a legend of
the representation are given in \autoref{fig:example-and-legend}. Due to the encoding of
numerical values as grayscale PNG images (allowable values are integers between 0 and
255), a small round-off error is present during a single round of encoding and decoding.
An example comparing an original vs. decoded structure is given in
\autoref{fig:original-decoded}.

![(a) Original and (b) `xtal2png` decoded visualizations of
[`mp-560471`](https://materialsproject.org/materials/mp-560471/) / $Zn_2B_2PbO_6$. Images were generated via [`pymatviz`](https://github.com/janosh/pymatviz) [@riebesellPymatviz2022]. \label{fig:original-decoded}](figures/original-decoded.png){ width=50% }

The significance of the representation lies in being able to directly use the PNG
representation with image-based models which often do not directly support custom
dataset types, potentially saving days or weeks during the process of obtaining
preliminary results on a newly released model.

We plan to apply `xtal2png` to a probabilistic diffusion generative model as a
proof-of-concept and present our findings in the near future.

<!-- ![Caption for example figure.\label{fig:example}](figure.png) -->

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
