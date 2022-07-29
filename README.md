[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![ReadTheDocs](https://readthedocs.org/projects/xtal2png/badge/?version=latest)](https://xtal2png.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/sparks-baird/xtal2png/main.svg)](https://coveralls.io/r/sparks-baird/xtal2png)
[![PyPI-Server](https://img.shields.io/pypi/v/xtal2png.svg)](https://pypi.org/project/xtal2png/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/xtal2png.svg)](https://anaconda.org/conda-forge/xtal2png)
![Lines of code](https://img.shields.io/tokei/lines/github/sparks-baird/xtal2png)
[![status](https://joss.theoj.org/papers/0c704f6ae9739c1e97e05ae0ad57aecb/status.svg)](https://joss.theoj.org/papers/0c704f6ae9739c1e97e05ae0ad57aecb)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/xtal2png)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/xtal2png.svg?branch=main)](https://cirrus-ci.com/github/<USER>/xtal2png)
[![Monthly Downloads](https://pepy.tech/badge/xtal2png/month)](https://pepy.tech/project/xtal2png)
-->

> ⚠️ Manuscript and results using a generative model coming soon ⚠️

# xtal2png [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sparks-baird/xtal2png/blob/main/notebooks/1.0-xtal2png-tutorial.ipynb)

> Encode/decode a crystal structure to/from a grayscale PNG image for direct use with image-based machine learning models such as [Google's Imagen](https://imagen.research.google/).

The latest advances in machine learning are often in natural language such as with LSTMs and transformers or image processing such as with GANs, VAEs, and guided diffusion models. Encoding/decoding crystal structures via grayscale PNG images is akin to making/reading a QR code for crystal structures. This allows you, as a materials informatics practitioner, to get streamlined results for new state-of-the-art image-based machine learning models applied to crystal structure. Let's take Google's text-to-image diffusion model, [Imagen](https://imagen.research.google/) ([unofficial](https://github.com/lucidrains/imagen-pytorch)), which [can also be used as an image-to-image model](https://github.com/lucidrains/imagen-pytorch#usage). Rather than dig into the code spending hours, days, or weeks modifying, debugging, and playing GitHub phone tag with the developers before you can (maybe) get preliminary results, `xtal2png` lets you get those results using the default instructions on the repository.

After getting preliminary results, you get to decide whether it's worth it to you to take on the higher-cost/higher-expertise task of modifying the codebase and using a more customized approach. Or, you can stick with the results of `xtal2png`. It's up to you!

## Getting Started

### Installation

```bash
conda create -n xtal2png -c conda-forge xtal2png m3gnet
conda activate xtal2png
```
> NOTE: [`m3gnet`](https://github.com/materialsvirtuallab/m3gnet) is an optional dependency that performs surrogate DFT relaxation.

### Example

Here, we use the top-level
[`XtalConverter`](https://xtal2png.readthedocs.io/en/latest/api/xtal2png.html#xtal2png.core.XtalConverter)
class with and without optional relaxation via
[`m3gnet`](https://github.com/materialsvirtuallab/m3gnet).

```python
# example_structures is a list of `pymatgen.core.structure.Structure` objects
>>> from xtal2png import XtalConverter, example_structures
>>>
>>> xc = XtalConverter(relax_on_decode=False)
>>> data = xc.xtal2png(example_structures, show=True, save=True)
>>> decoded_structures = xc.png2xtal(data, save=False)
>>> len(decoded_structures)
2

>> xc = XtalConverter(relax_on_decode=True)
>> data = xc.xtal2png(example_structures, show=True, save=True)
>> relaxed_decoded_structures = xc.png2xtal(data, save=False)
>> len(relaxed_decoded_structures)
2

```



### Output

```python
print(example_structures[0], decoded_structures[0], relaxed_decoded_structures[0])
```



<table>
<tr>
<th> Original </th>
</tr>
<tr>
<td>

```python
Structure Summary
Lattice
    abc : 5.033788 11.523021 10.74117
 angles : 90.0 90.0 90.0
 volume : 623.0356027127609
      A : 5.033788 0.0 3.0823061808931787e-16
      B : 1.8530431062799525e-15 11.523021 7.055815392078867e-16
      C : 0.0 0.0 10.74117
PeriodicSite: Zn2+ (0.9120, 5.7699, 9.1255) [0.1812, 0.5007, 0.8496]
PeriodicSite: Zn2+ (4.1218, 5.7531, 1.6156) [0.8188, 0.4993, 0.1504]
...
```
</td>
</table>

<table>
<tr>
<th> Decoded </th>
</tr>
<tr>
<td>

```python
Structure Summary
Lattice
    abc : 5.0250980392156865 11.533333333333331 10.8
 angles : 90.0 90.0 90.0
 volume : 625.9262117647058
      A : 5.0250980392156865 0.0 0.0
      B : 0.0 11.533333333333331 0.0
      C : 0.0 0.0 10.8
PeriodicSite: Zn (0.9016, 5.7780, 3.8012) [0.1794, 0.5010, 0.3520]
PeriodicSite: Zn (4.1235, 5.7554, 6.9988) [0.8206, 0.4990, 0.6480]
...
```

</td>
</table>

<table>
<tr>
<th> Relaxed Decoded </th>
</tr>
<tr>
<td>

```python
Structure Summary
Lattice
    abc : 5.026834307381214 11.578854613685237 10.724087971087924
 angles : 90.0 90.0 90.0
 volume : 624.1953646135236
      A : 5.026834307381214 0.0 0.0
      B : 0.0 11.578854613685237 0.0
      C : 0.0 0.0 10.724087971087924
PeriodicSite: Zn (0.9050, 5.7978, 3.7547) [0.1800, 0.5007, 0.3501]
PeriodicSite: Zn (4.1218, 5.7810, 6.9693) [0.8200, 0.4993, 0.6499]
...
```

</td>
</tr>
</table>

The before and after structures match within an expected tolerance; note the round-off error due to encoding numerical data as RGB images which has a coarse resolution of approximately `1/255 = 0.00392`. Note also that the decoded version lacks charge states. The QR-code-like intermediate PNG image is also provided in original size and a scaled version for a better viewing experience:
| 64x64 pixels                                                                                                                              | Scaled for Better Viewing ([tool credit](https://lospec.com/pixel-art-scaler/))                                           | Legend                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| ![Zn8B8Pb4O24,volume=623,uid=bc2d](https://user-images.githubusercontent.com/45469701/169936372-e14a8bba-698a-4fc9-9d4b-fc5e1de7d67f.png) | <img src=https://user-images.githubusercontent.com/45469701/169936297-57f5afb6-c4ae-4d8a-8cbb-33dcaf190b98.png width=400> | <img src=https://user-images.githubusercontent.com/45469701/169937021-f6f60169-6965-4db1-9bbd-e8744521d570.png width=400> |

Additional examples can be found in [the docs](https://xtal2png.readthedocs.io/en/latest/examples.html).

## Limitations and Design Considerations

There are some limitations and design considerations for `xtal2png`. Here, we cover round-off error, image dimensions, contextual features, and customization.

### Round-off
While the round-off
error is a necessary evil for encoding to a [PNG file format](https://en.wikipedia.org/wiki/Portable_Network_Graphics), the unrounded NumPy arrays
can be used directly instead if supported by the image model of interest via
[`structures_to_arrays`](https://xtal2png.readthedocs.io/en/latest/api/xtal2png.html#xtal2png.core.XtalConverter.structures_to_arrays) and [`arrays_to_structures`](https://xtal2png.readthedocs.io/en/latest/api/xtal2png.html#xtal2png.XtalConverter.arrays_to_structures).

### Image dimensions
We choose a
$64\times64$ representation by default which supports up to 52 sites within a unit cell.
The maximum number of sites [`max_sites`](https://xtal2png.readthedocs.io/en/latest/api/xtal2png.html#xtal2png.core.XtalConverter) can be adjusted which changes the size of the
representation. A square representation is used for greater compatibility with the
common limitation of image-based models supporting only square image arrays. The choice
of the default sidelength as a base-2 number (i.e. $2^6$) reflects common conventions of
low-resolution images for image-based machine learning tasks.

### Contextual features
While the
distance matrix does not directly contribute to the reconstruction in the current
implementation of `xtal2png`, it serves a number of purposes. First, similar to the unit
cell volume and space group information, it can provide additional guidance to the
algorithm. A corresponding example would be the role of background vs. foreground in
classification of wolves vs. huskies; oftentimes classification algorithms will pay
attention to the background (such as presence of snow) in predicting the animal class.
Likewise, providing contextual information such as volume, space group, and a distance
matrix is additional information that can help the models to capture the essence of
particular crystal structures. In a future implementation, we plan to reconstruct
Euclidean coordinates from the distance matrices and homogenize (e.g. via weighted
averaging) the explicit fractional coordinates with the reconstructed coordinates.

### Customization
See the [docs](https://xtal2png.readthedocs.io/en/latest/api/xtal2png.html#xtal2png.core.XtalConverter) for the full list of customizable parameters that `XtalConverter` takes.

## Installation

<!-- ### Anaconda (`conda`) installation (recommended)

(2022-05-23, conda-forge installation still pending, fallback to `pip install xtal2png` as separate command)

Create and activate a new `conda` environment named `xtal2png` (`-n`) that will search for and install the `xtal2png` package from the `conda-forge` Anaconda channel (`-c`).

```bash
conda env create -n xtal2png -c conda-forge xtal2png
conda activate xtal2png
```

Alternatively, in an already activated environment:

```bash
conda install -c conda-forge xtal2png
```

If you run into conflicts with packages you are integrating with `xtal2png`, please try installing all packages in a single line of code (or two if mixing `conda` and `pip` packages in the same environment) and/or installing with `mamba` ([source](https://stackoverflow.com/a/69137255/13697228)). -->

### PyPI (`pip`) installation

Create and activate a new `conda` environment named `xtal2png` (`-n`) with `python==3.9.*` or your preferred Python version, then install `xtal2png` via `pip`.

```bash
conda create -n xtal2png python==3.9.*
conda activate xtal2png
pip install xtal2png
```

## Editable installation

In order to set up the necessary environment:

1. clone and enter the repository via:

   ```bash
   git clone https://github.com/sparks-baird/xtal2png.git
   cd xtal2png
   ```

2. create and activate a new conda environment (optional, but recommended)

   ```bash
   conda env create --name xtal2png python==3.9.*
   conda activate xtal2png
   ```

3. perform an editable (`-e`) installation in the current directory (`.`):

   ```bash
   pip install -e .
   ```

> **_NOTE:_**  Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.

Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:

   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```

   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:

   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```

   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.

<!-- ## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n xtal2png -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ``` -->

## Command Line Interface (CLI)

Make sure to install the package first per the installation instructions above. Here is
how to access the help for the CLI and a few examples to get you started.

### Help

You can see the usage information of the `xtal2png` CLI script via:

```bash
xtal2png --help
```

> ```bash
>Usage: xtal2png [OPTIONS]
>
>  xtal2png command line interface.
>
>Options:
>  --version                 Show version.
>  -p, --path PATH           Crystallographic information file (CIF) filepath
>                            (extension must be .cif or .CIF) or path to
>                            directory containing .cif files or processed PNG
>                            filepath or path to directory containing processed
>                            .png files (extension must be .png or .PNG).
>                            Assumes CIFs if --encode flag is used. Assumes
>                            PNGs if --decode flag is used.
>  -s, --save-dir PATH       Encode CIF files as PNG images.
>  --encode                  Encode CIF files as PNG images.
>  --decode                  Decode PNG images to CIF files.
>  -v, --verbose TEXT        Set loglevel to INFO.
>  -vv, --very-verbose TEXT  Set loglevel to INFO.
>  --help                    Show this message and exit.
> ```

### Examples

To encode a single CIF file located at `src/xtal2png/utils/Zn2B2PbO6.cif` as a PNG and save the PNG to the `tmp` directory:

```bash
xtal2png --encode --path src/xtal2png/utils/Zn2B2PbO6.cif --save-dir tmp
```

To encode all CIF files contained in the `src/xtal2png/utils` directory as a PNG and
save corresponding PNGs to the `tmp` directory:

```bash
xtal2png --encode --path src/xtal2png/utils --save-dir tmp
```

To decode a single structure-encoded PNG file located at
`data/preprocessed/Zn8B8Pb4O24,volume=623,uid=b62a.png` as a CIF file and save the CIF
file to the `tmp` directory:

```bash
xtal2png --decode --path data/preprocessed/Zn8B8Pb4O24,volume=623,uid=b62a.png --save-dir tmp
```

To decode all structure-encoded PNG file contained in the `data/preprocessed` directory as CIFs and save the CIFs to the `tmp` directory:

```bash
xtal2png --decode --path data/preprocessed --save-dir tmp
```

Note that the save directory (e.g. `tmp`) including any parents (e.g. `ab/cd/tmp`) will
be created automatically if the directory does not already exist.

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── preprocessed        <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── xtal2png            <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note on PyScaffold

This project has been set up using [PyScaffold] 4.2.1 and the [dsproject extension] 0.7.1.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject

To create the same starting point for this repository, as of 2022-06-01 on Windows you will need the development versions of PyScaffold and extensions, however this will not be necessary once certain bugfixes have been introduced in the next stable releases:

```bash
pip install git+https://github.com/pyscaffold/pyscaffold.git git+https://github.com/pyscaffold/pyscaffoldext-dsproject.git git+https://github.com/pyscaffold/pyscaffoldext-markdown.git
```

The following `pyscaffold` command creates a starting point for this repository:

```bash
putup xtal2png --github-actions --markdown --dsproj
```

Alternatively, you can edit a file interactively and update and uncomment relevant lines, which saves some of the additional setup:

```bash
putup --interactive xtal2png
```

## Attributions

- [@michaeldalverson](https://github.com/michaeldalverson) for iterating through various representations during extensive work with crystal GANs. The base representation for `xtal2png` (see [#output](https://github.com/sparks-baird/xtal2png#output)) closely follows a recent iteration (2022-06-13), taking the first layer ($1\times64\times64$) of the $4\times64\times64$ representation and replacing a buffer column/row of zeros with unit cell volume.
