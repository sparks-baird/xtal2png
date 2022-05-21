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
from uuid import uuid4

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
    feature_range : Sequence
        The scaled values will span the range of ``feature_range``
    data_range : Sequence
        Expected bounds for the data, e.g. 0 to 117 for periodic elements

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


def element_wise_unscaler(
    X_scaled: ArrayLike,
    feature_range: Sequence,
    data_range: Sequence,
):
    """Scale parameters according to a prespecified min and max (``data_range``).

    ``feature_range`` is preserved from MinMaxScaler

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Element-wise scaled values.
    feature_range : Sequence
        The scaled values will span the range of ``feature_range``
    data_range : Sequence
        Expected bounds for the data, e.g. 0 to 117 for periodic elements

    Returns
    -------
    X
        Element-wise unscaled values.
    """
    if not isinstance(X_scaled, np.ndarray):
        X_scaled = np.array(X_scaled)

    data_min, data_max = data_range
    feature_min, feature_max = feature_range
    # following modified from:
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    # inverse transform, checked against Mathematica
    X_std = (X_scaled - feature_min) / (feature_max - feature_min)
    X = data_min + (data_max - data_min) * X_std
    return X


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


def rgb_unscaler(
    X: ArrayLike,
    data_range: Optional[Sequence] = None,
):
    """Unscale parameters from their RGB scale (0 to 255).

    ``feature_range`` is fixed to [0, 255], ``data_range`` is either specified or
    calculated based on min and max.

    See Also
    --------
    sklearn.preprocessing.MinMaxScaler : Scale each feature to a given range.

    Parameters
    ----------
    X : ArrayLike
        Element-wise scaled values.
    data_range : Optional[Sequence]
        Range to use in place of np.min(X) and np.max(X) as in ``class:MinMaxScaler``.

    Returns
    -------
    X
        Unscaled features.

    Examples
    --------
    >>> rgb_unscaler([[32, 64], [96, 128]], data_range=[0, 8])
    array([[1, 2],
          [3, 4]])
    """
    rgb_range = [0, 255]
    X_scaled = element_wise_unscaler(X, data_range=data_range, feature_range=rgb_range)
    return X_scaled


class XtalConverter:
    """Convert between pymatgen Structure object and PNG-encoded representation."""

    def __init__(
        self,
        atom_range: Tuple[int, int] = (0, 117),
        frac_range: Tuple[float, float] = (0.0, 1.0),
        abc_range: Tuple[float, float] = (0.0, 10.0),
        angles_range: Tuple[float, float] = (0.0, 180.0),
        volume_range: Tuple[float, float] = (0.0, 1000.0),
        space_group_range: Tuple[int, int] = (1, 230),
        distance_range: Tuple[float, float] = (0.0, 25.0),
        max_sites: int = 52,
        save_dir: Union[str, PathLike[str]] = path.join("data", "preprocessed"),
    ):
        """Instantiate an XtalConverter object with desired ranges and ``max_sites``.

        Parameters
        ----------
        atom_range : Tuple[int, int], optional
            Expected range for atomic number, by default (0, 117)
        frac_range : Tuple[float, float], optional
            Expected range for fractional coordinates, by default (0.0, 1.0)
        abc_range : Tuple[float, float], optional
            Expected range for lattice parameter lengths, by default (0.0, 10.0)
        angles_range : Tuple[float, float], optional
            Expected range for lattice parameter angles, by default (0.0, 180.0)
        volume_range : Tuple[float, float], optional
            Expected range for unit cell volumes, by default (0.0, 1000.0)
        space_group_range : Tuple[int, int], optional
            Expected range for space group numbers, by default (1, 230)
        distance_range : Tuple[float, float], optional
            Expected range for pairwise distances between sites, by default (0.0, 25.0)
        max_sites : int, optional
            Maximum number of sites to accomodate in encoding, by default 52
        save_dir : Union[str, PathLike[str]]
            Directory to save PNG files via ``func:xtal2png``,
            by default path.join("data", "interim")
        """
        self.atom_range = atom_range
        self.frac_range = frac_range
        self.abc_range = abc_range
        self.angles_range = angles_range
        self.volume_range = volume_range
        self.space_group_range = space_group_range
        self.distance_range = distance_range
        self.max_sites = max_sites
        self.save_dir = save_dir

    def xtal2png(
        self,
        structures: List[Union[Structure, str, PathLike[str]]],
        show: bool = False,
        save: bool = True,
        gen_mask: bool = False,
        comp_mask: bool = False,
    ):
        """Encode crystal (via CIF filepath or Structure object) as PNG file.

        Parameters
        ----------
        structures : List[Union[Structure, str, PathLike[str]]]
            pymatgen Structure objects or path to CIF files.
        show : bool, optional
            Whether to display the PNG-encoded file, by default False
        save : bool, optional
            Whether to save the PNG-encoded file, by default True

        Returns
        -------
        imgs : List[Image.Image]
            PIL images that (approximately) encode the supplied crystal structures.

        Raises
        ------
        ValueError
            structures should be of same datatype
        ValueError
            structures should be of same datatype
        ValueError
            structures should be of type `str`, `os.PathLike` or
            `pymatgen.core.structure.Structure`

        Examples
        --------
        >>> coords = [[0, 0, 0], [0.75,0.5,0.75]]
        >>> lattice = Lattice.from_parameters(
        ...     a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60
        ... )
        >>> structures = [Structure(lattice, ["Si", "Si"], coords),
        ... Structure(lattice, ["Ni", "Ni"], coords)]
        >>> xc = XtalConverter()
        >>> xc.xtal2png(structures, show=False, save=True)
        """
        save_names, S = self.process_filepaths(structures)

        # convert structures to 3D NumPy Matrices
        self.data, self.id_data, self.id_keys = self.structures_to_arrays(S)

        if gen_mask:

            self.data[self.id_data == self.id_keys]

        # convert to PNG images. Save and/or show, if applicable
        imgs: List[Image.Image] = []
        for d, save_name in zip(self.data, save_names):
            img = Image.fromarray(d, mode="L")
            imgs.append(img)
            if save:
                savepath = path.join(self.save_dir, save_name + ".png")
                img.save(savepath)
            if show:
                img.show()

        return imgs

    def process_filepaths(self, structures):
        save_names: List[str] = []
        S: List[Structure] = []
        first_is_structure = isinstance(structures[0], Structure)
        for i, s in enumerate(structures):
            if isinstance(s, str) or isinstance(s, PathLike):
                if first_is_structure:
                    raise ValueError(
                        f"structures should be of same datatype, either strs or pymatgen Structures. structures[0] is {type(structures[0])}, but got type {type(s)} for entry {i}"  # noqa
                    )

                # load the CIF and convert to a pymatgen Structure
                S.append(Structure.from_file(s))
                save_names.append(str(s))

            elif isinstance(s, Structure):
                if not first_is_structure:
                    raise ValueError(
                        f"structures should be of same datatype, either strs or pymatgen Structures. structures[0] is {type(structures[0])}, but got type {type(s)} for entry {i}"  # noqa
                    )

                S.append(s)
                save_names.append(
                    f"{s.formula.replace(' ', '')},volume={int(np.round(s.volume))},uid={str(uuid4())[0:4]}"  # noqa
                )
            else:
                raise ValueError(
                    f"structures should be of type `str`, `os.PathLike` or `pymatgen.core.structure.Structure`, not {type(S)} (entry {i})"  # noqa
                )

        return save_names, S

    def png2xtal(self, images: List[Union[Image.Image, PathLike]], save: bool = False):
        """_summary_

        Parameters
        ----------
        images : List[Union[Image.Image, PathLike]]
            PIL images that (approximately) encode crystal structures.

        Examples
        --------
        >>> png2xtal(images)
        OUTPUT
        """
        data_tmp = []
        for img in images:
            if isinstance(img, str):
                # load image from file
                with Image.open(img) as im:
                    data_tmp.append(np.asarray(im))
            elif isinstance(img, Image.Image):
                data_tmp.append(np.asarray(img))

        data = np.stack(data_tmp, axis=2)

        S = self.arrays_to_structures(data)

        if save:
            # save new CIF files
            1 + 1

        return S

        # unscale values

    def structures_to_arrays(self, structures: Sequence[Structure]):
        """Convert pymatgen Structure to scaled 3D array of crystallographic info.

        ``atomic_numbers`` and ``distance_matrix` get padded or cropped as appropriate,
        as these depend on the number of sites in the structure.

        Parameters
        ----------
        S : Sequence[Structure]
            Sequence (e.g. list) of pymatgen Structure object(s)
        """
        if isinstance(structures, Structure):
            raise ValueError("`structures` should be a list of pymatgen Structure(s)")

        # extract crystallographic information
        atomic_numbers: List[List[int]] = []
        frac_coords_tmp: List[NDArray] = []
        abc: List[List[float]] = []
        angles: List[List[float]] = []
        volume: List[float] = []
        space_group: List[int] = []
        distance_matrix_tmp: List[NDArray[np.float64]] = []

        for s in structures:
            n_sites = len(s.atomic_numbers)
            if n_sites > self.max_sites:
                raise ValueError(
                    "crystal supplied with {n_sites} sites, which is more than {max_sites} sites. Remove crystal or increase `max_sites`."  # noqa
                )
            atomic_numbers.append(
                np.pad(
                    list(s.atomic_numbers),
                    (0, self.max_sites - n_sites),
                ).tolist()
            )
            frac_coords_tmp.append(
                np.pad(s.frac_coords, ((0, self.max_sites - n_sites), (0, 0)))
            )
            abc.append(list(s._lattice.abc))
            angles.append(list(s._lattice.angles))
            volume.append(s.volume)
            space_group.append(s.get_space_group_info()[1])

            if n_sites != s.distance_matrix.shape[0]:
                raise ValueError(
                    f"len(atomic_numbers) {n_sites} and distance_matrix.shape[0] {s.distance_matrix.shape[0]} do not match"  # noqa
                )  # noqa

            # assume that distance matrix is square
            padwidth = (0, self.max_sites - n_sites)
            distance_matrix_tmp.append(np.pad(s.distance_matrix, padwidth))
            # [0:max_sites, 0:max_sites]

        frac_coords = np.stack(frac_coords_tmp)
        distance_matrix = np.stack(distance_matrix_tmp)

        # REVIEW: consider using modified pettifor scale instead of atomic numbers
        # REVIEW: consider using feature_range=atom_range or 2*atom_range
        # REVIEW: since it introduces a sort of non-linearity b.c. of rounding
        atom_scaled = rgb_scaler(atomic_numbers, data_range=self.atom_range)
        frac_scaled = rgb_scaler(frac_coords, data_range=self.frac_range)
        abc_scaled = rgb_scaler(abc, data_range=self.abc_range)
        angles_scaled = rgb_scaler(angles, data_range=self.angles_range)
        volume_scaled = rgb_scaler(volume, data_range=self.volume_range)
        space_group_scaled = rgb_scaler(space_group, data_range=self.space_group_range)
        # NOTE: max_distance could be added as another (repeated) value/row to infer
        # NOTE: kind of like frac_distance_matrix, not sure if would be effective
        # NOTE: Or could normalize distance_matix by cell volume
        # NOTE: and possibly include cell volume as a (repeated) value/row to infer
        # NOTE: It's possible extra info like this isn't so bad, instilling the physics
        # NOTE: but it could also just be extraneous work to predict/infer
        distance_scaled = rgb_scaler(distance_matrix, data_range=self.distance_range)

        # atom_scaled = np.array(atomic_numbers)
        # frac_scaled = frac_coords
        # abc_scaled = np.array(abc)
        # angles_scaled = np.array(angles)
        # volume_scaled = np.array(volume)
        # space_group_scaled = np.array(space_group)
        # distance_scaled = distance_matrix

        atom_arr = np.expand_dims(atom_scaled, 2)
        frac_arr = frac_scaled
        abc_arr = np.repeat(np.expand_dims(abc_scaled, 1), self.max_sites, axis=1)
        angles_arr = np.repeat(np.expand_dims(angles_scaled, 1), self.max_sites, axis=1)
        volume_arr = np.repeat(
            np.expand_dims(volume_scaled, (1, 2)), self.max_sites, axis=1
        )
        space_group_arr = np.repeat(
            np.expand_dims(space_group_scaled, (1, 2)), self.max_sites, axis=1
        )
        distance_arr = distance_scaled

        data = self.assemble_blocks(
            atom_arr,
            frac_arr,
            abc_arr,
            angles_arr,
            volume_arr,
            space_group_arr,
            distance_arr,
        )

        ATOM_ID = 0
        FRAC_ID = 1
        ABC_ID = 2
        ANGLES_ID = 3
        VOLUME_ID = 4
        SPACE_GROUP_ID = 5
        DISTANCE_ID = 6

        id_keys = dict(
            atom=ATOM_ID,
            frac=FRAC_ID,
            abc=ABC_ID,
            angles=ANGLES_ID,
            volume=VOLUME_ID,
            space_group=SPACE_GROUP_ID,
            distance=DISTANCE_ID,
        )
        id_blocks = [
            np.ones_like(atom_arr) * ATOM_ID,
            np.ones_like(frac_arr) * FRAC_ID,
            np.ones_like(abc_arr) * ABC_ID,
            np.ones_like(angles_arr) * ANGLES_ID,
            np.ones_like(volume_arr) * VOLUME_ID,
            np.ones_like(space_group_arr) * SPACE_GROUP_ID,
            np.ones_like(distance_arr) * DISTANCE_ID,
        ]
        id_data = self.assemble_blocks(*id_blocks)

        return data, id_data, id_keys

    def assemble_blocks(
        self,
        atom_arr,
        frac_arr,
        abc_arr,
        angles_arr,
        volume_arr,
        space_group_arr,
        distance_arr,
    ):
        arrays = [
            atom_arr,
            frac_arr,
            abc_arr,
            angles_arr,
            volume_arr,
            space_group_arr,
        ]
        zero_pad = sum([arr.shape[2] for arr in arrays])
        zero = np.zeros((2, zero_pad, zero_pad), dtype=np.uint8)

        vertical_arr = np.block(
            [
                [zero],
                [
                    atom_arr,
                    frac_arr,
                    abc_arr,
                    angles_arr,
                    volume_arr,
                    space_group_arr,
                ],
            ]
        )
        horizontal_arr = np.block(
            [atom_arr, frac_arr, abc_arr, angles_arr, volume_arr, space_group_arr]
        )
        horizontal_arr = np.moveaxis(horizontal_arr, 1, 2)
        left_arr = vertical_arr
        right_arr = np.block([[horizontal_arr], [distance_arr]])
        data = np.block([left_arr, right_arr])
        return data

    def arrays_to_structures(cls, data: np.ndarray):
        """Convert scaled 3D crystal (xtal) array to pymatgen Structure.

        Parameters
        ----------
        data : np.ndarray
            3D array containing crystallographic information.
        """
        for d in data:
            1
            # extract individual parts (opposite of np.block)
        # round fractional coordinates to nearest multiple?

        # average repeated values

        # build Structure-s


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

# (
#     atom_scaled,
#     frac_scaled,
#     abc_scaled,
#     angles_scaled,
#     space_group_scaled,
#     distance_scaled,
# )
# id_blocks = [b * i for i, b in enumerate(id_blocks)]
