"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = xtal2png.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This skeleton file can be safely removed if not needed!

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

# from itertools import zip_longest
from os import PathLike, path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image
from pymatgen.core.structure import Structure

from xtal2png import __version__

# from sklearn.preprocessing import MinMaxScaler


__author__ = "sgbaird"
__copyright__ = "sgbaird"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from xtal2png.skeleton import fib`,
# when using this Python module as a library.


# def fib(n):
#     """Fibonacci example function

#     Args:
#       n (int): integer

#     Returns:
#       int: n-th Fibonacci number
#     """
#     assert n > 0
#     a, b = 1, 1
#     for _i in range(n - 1):
#         a, b = b, a + b
#     return a


def element_wise_scaler(
    X: ArrayLike,
    feature_range: Optional[Sequence] = None,
    data_range: Optional[Sequence] = None,
):
    """Scale parameters according to a prespecified min and max (``data_range``).

    ``feature_range`` is preserved from MinMaxScaler

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.

    Returns
    -------
    X_scaled
        Element-wise scaled values.
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if data_range is None:
        data_range = [np.min(X), np.max(X)]
    if feature_range is None:
        feature_range = [np.min(X), np.max(X)]

    data_min, data_max = data_range
    feature_min, feature_max = feature_range
    # following modified from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    X_std = (X - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min
    return X_scaled


def rgb_scaler(
    X: ArrayLike,
    data_range: Optional[Sequence] = None,
):
    """Scale parameters according to RGB scale (0 to 255).

    ``feature_range`` is fixed to [0, 255], ``data_range`` is either specified

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Features to be scaled element-wise.
    data_range : Optional[Sequence]
        Range to use in place of np.min(X) and np.max(X) as in ``MinMaxScaler``.

    Returns
    -------
    X_scaled
        Element-wise scaled values.

    Examples
    --------
    >>> rgb_scaler([[1, 2], [3, 4]], data_range=[0, 8])
    array([[ 32,  64],
        [ 96, 128]])
    """
    rgb_range = [0, 255]
    X_scaled = element_wise_scaler(X, data_range=data_range, feature_range=rgb_range)
    X_scaled = np.round(X_scaled).astype(np.uint8)
    return X_scaled


class XtalConverter:
    @classmethod
    def xtal2png(
        cls,
        structure: Union[Structure, str, PathLike[str]],
        savedir: Union[str, PathLike[str]] = path.join("data", "interim"),
        savename: str = "tmp",
        show: bool = False,
    ):
        if isinstance(structure, str) or isinstance(structure, PathLike):
            # load the CIF and convert to a pymatgen Structure
            S = Structure.from_file(structure)
        elif isinstance(structure, Structure):
            S = structure
        else:
            raise ValueError(
                f"structure should be of type `str`, `os.PathLike` or `pymatgen.core.structure.Structure`, not {type(S)}"  # noqa
            )

        # convert `S` to 3D NumPy Matrix
        data = np.random.rand(64, 64, 3)

        # scale values

        # convert to a PNG image and save
        img = Image.fromarray(data, mode="RGB")
        savepath = path.join(savedir, savename + ".png")
        img.save(savepath)
        if show:
            img.show()

        return savepath

    @classmethod
    def png2xtal(cls, image: Union[Image.Image, PathLike]):
        """_summary_

        Parameters
        ----------
        image : Union[Image.Image, PathLike]
            _description_

        Examples
        --------
        >>> png2xtal(image, , )
        OUTPUT
        """
        if isinstance(image, str):
            # load image from file
            with Image.open(image) as im:
                data = np.asarray(im)
        elif isinstance(image, Image.Image):
            data = np.asarray(image)

        data

        # unscale values

    @classmethod
    def structures_to_arrays(
        cls,
        structures: Sequence[Structure],
        atom_range: Tuple[int, int] = (0, 117),
        frac_range: Tuple[float, float] = (0.0, 1.0),
        abc_range: Tuple[float, float] = (0.0, 10.0),
        angles_range: Tuple[float, float] = (0.0, 180.0),
        space_group_range: Tuple[int, int] = (1, 230),
        distance_range: Tuple[float, float] = (0.0, 25.0),
    ):
        """Convert pymatgen Structure to scaled 3D array of crystallographic info.

        Parameters
        ----------
        S : Sequence[Structure]
            Sequence (e.g. list) of pymatgen Structure objects
        """
        # extract crystallographic information
        atomic_numbers: List[List[int]] = []
        frac_coords_tmp: List[NDArray] = []
        abc: List[List[float]] = []
        angles: List[List[float]] = []
        space_group: List[int] = []
        distance_matrix_tmp: List[NDArray[np.float64]] = []

        max_sites = 52

        for s in structures:
            atomic_numbers.append(
                np.pad(
                    list(s.atomic_numbers), (0, max_sites - len(s.atomic_numbers))
                ).tolist()
            )
            frac_coords_tmp.append(s.frac_coords)
            abc.append(list(s._lattice.abc))
            angles.append(list(s._lattice.angles))
            space_group.append(s.get_space_group_info()[1])
            # assumes that distance matrix is square
            padwidth = (0, max_sites - s.distance_matrix.shape[0])
            distance_matrix_tmp.append(np.pad(s.distance_matrix, padwidth))

        frac_coords = np.concatenate(frac_coords_tmp)
        distance_matrix = np.stack(distance_matrix_tmp)

        atom_scaled = rgb_scaler(atomic_numbers, data_range=atom_range)
        frac_scaled = rgb_scaler(frac_coords, data_range=frac_range)
        abc_scaled = rgb_scaler(abc, data_range=abc_range)
        angles_scaled = rgb_scaler(angles, data_range=angles_range)
        space_group_scaled = rgb_scaler(space_group, data_range=space_group_range)
        distance_scaled = rgb_scaler(distance_matrix, data_range=distance_range)

        (
            atom_scaled,
            frac_scaled,
            abc_scaled,
            angles_scaled,
            space_group_scaled,
            distance_scaled,
        )

    @classmethod
    def arrays_to_structures(cls, data: np.ndarray):
        """Convert scaled 3D crystal (xtal) array to pymatgen Structure.

        Parameters
        ----------
        data : np.ndarray
            3D array containing crystallographic information.
        """
        #


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Crystal to PNG converter.")
    parser.add_argument(
        "--version",
        action="version",
        version="xtal2png {ver}".format(ver=__version__),
    )
    parser.add_argument(dest="fpath", help="CIF filepath", type=str, metavar="STRING")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Beginning conversion to PNG format")
    print(f"The PNG file is saved at {XtalConverter.xtal2png(args.fpath)}")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m xtal2png.skeleton 42
    #
    run()


# %% Code Graveyard
# @classmethod
# def scale_array(data: np.ndarray):
#     """Scale the 3D xtal array values as preprocessing step for PNG format.

#     Parameters
#     ----------
#     data : np.ndarray
#         Unscaled 3D array of crystallographic information.

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     #

# @classmethod
# def unscale_array(data: np.ndarray):
#     """Unscale the 3D xtal array values as part of conversion from PNG to Structure.

#     Parameters
#     ----------
#     data : np.ndarray
#         Scaled 3D array of crystallographic information.

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     #

# atomic_numbers = list(zip_longest(*atomic_numbers, fillvalue=0))

# atom_scaler = MinMaxScaler(feature_range=atom_range)
# frac_scaler = MinMaxScaler(feature_range=frac_range)
# abc_scaler = MinMaxScaler(feature_range=abc_range)
# angles_scaler = MinMaxScaler(feature_range=angles_range)
# space_group_scaler = MinMaxScaler(feature_range=space_group_range)
# distance_scaler = MinMaxScaler(feature_range=distance_range)

# atom_scaled = atom_scaler.fit_transform(atomic_numbers)
# frac_scaled = frac_scaler.fit_transform(frac_coords)
# abc_scaled = abc_scaler.fit_transform(abc)
# angles_scaled = angles_scaler.fit_transform(angles)
# space_group_scaled = space_group_scaler.fit_transform(space_group)
# distance_scaled = distance_scaler.fit_transform(distance_matrix)

# atomic_numbers: NDArray[np.int_] = np.array([])
# frac_coords: NDArray[np.float] = np.array([])
# abc: NDArray[np.float] = np.array([])
# angles: NDArray[np.float] = np.array([])
# space_group: NDArray[np.int_] = np.array([])
# distance_matrix: NDArray[np.float] = np.array([])
