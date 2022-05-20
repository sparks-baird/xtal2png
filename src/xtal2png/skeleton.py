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
from os import PathLike, path
from typing import Union

import numpy as np
from PIL import Image
from pymatgen.core.structure import Structure

from xtal2png import __version__

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


class Converter:
    @classmethod
    def xtal2png(
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
    def png2xtal(image: Union[Image.Image, PathLike]):
        if isinstance(image, str):
            # load image from file
            with Image.open(image) as im:
                data = np.asarray(im)
        elif isinstance(image, Image.Image):
            data = np.asarray(image)

        data

        # unscale values

    @classmethod
    def structure_to_array(structure: Structure):
        """Convert pymatgen Structure to scaled 3D array of crystallographic info.

        Parameters
        ----------
        structure : Structure
            _description_
        """
        # TODO: add parameters for min/max bounds for various things

    @classmethod
    def array_to_structure(data: np.ndarray):
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
    """Parse command line parameters

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
    print(f"The PNG file is saved at {Converter.xtal2png(args.fpath)}")
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
