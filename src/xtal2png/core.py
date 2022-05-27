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
from glob import glob

# from itertools import zip_longest
from os import PathLike, path
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from uuid import uuid4
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from xtal2png import __version__
from xtal2png.utils.data import dummy_structures, rgb_scaler, rgb_unscaler

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


ATOM_ID = 1
FRAC_ID = 2
ABC_ID = 3
ANGLES_ID = 4
VOLUME_ID = 5
SPACE_GROUP_ID = 6
DISTANCE_ID = 7

ATOM_KEY = "atom"
FRAC_KEY = "frac"
ABC_KEY = "abc"
ANGLES_KEY = "angles"
VOLUME_KEY = "volume"
SPACE_GROUP_KEY = "space_group"
DISTANCE_KEY = "distance"


class XtalConverter:
    """Convert between pymatgen Structure object and PNG-encoded representation."""

    def __init__(
        self,
        atom_range: Tuple[int, int] = (0, 117),
        frac_range: Tuple[float, float] = (0.0, 1.0),
        abc_range: Tuple[float, float] = (0.0, 15.0),
        angles_range: Tuple[float, float] = (0.0, 180.0),
        volume_range: Tuple[float, float] = (0.0, 1000.0),
        space_group_range: Tuple[int, int] = (1, 230),
        distance_range: Tuple[float, float] = (0.0, 25.0),
        max_sites: int = 52,
        save_dir: Union[str, "PathLike[str]"] = path.join("data", "preprocessed"),
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
        save_dir : Union[str, 'PathLike[str]']
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

        Path(save_dir).mkdir(exist_ok=True, parents=True)

    def xtal2png(
        self,
        structures: List[Union[Structure, str, "PathLike[str]"]],
        show: bool = False,
        save: bool = True,
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
        save_names, S = self.process_filepaths_or_structures(structures)

        # convert structures to 3D NumPy Matrices
        self.data, self.id_data, self.id_keys = self.structures_to_arrays(S)
        mn, mx = self.data.min(), self.data.max()
        if mn < 0:
            warn(
                f"lower RGB value(s) OOB ({mn} less than 0). thresholding to 0.. may throw off crystal structure parameters (e.g. if lattice parameters are thresholded)"  # noqa: E501
            )  # noqa
            self.data[self.data < 0] = 0

        if mx > 255:
            warn(
                f"upper RGB value(s) OOB ({mx} greater than 255). thresholding to 255.. may throw off crystal structure parameters (e.g. if lattice parameters are thresholded)"  # noqa: E501
            )  # noqa
            self.data[self.data > 255] = 255

        self.data = self.data.astype(np.uint8)

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

    def process_filepaths_or_structures(self, structures):
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

    def png2xtal(
        self, images: List[Union[Image.Image, "PathLike"]], save: bool = False
    ):
        """_summary_

        Parameters
        ----------
        images : List[Union[Image.Image, 'PathLike']]
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

        data = np.stack(data_tmp, axis=0)

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

        id_mapper = {
            ATOM_KEY: ATOM_ID,
            FRAC_KEY: FRAC_ID,
            ABC_KEY: ABC_ID,
            ANGLES_KEY: ANGLES_ID,
            VOLUME_KEY: VOLUME_ID,
            SPACE_GROUP_KEY: SPACE_GROUP_ID,
            DISTANCE_KEY: DISTANCE_ID,
        }

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

        return data, id_data, id_mapper

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
        n_structures = atom_arr.shape[0]
        zero = np.zeros((n_structures, zero_pad, zero_pad), dtype=int)

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

    def disassemble_blocks(
        self, data, id_data: Optional[NDArray] = None, id_mapper: Optional[dict] = None
    ):
        if (id_data is None) is not (id_mapper is None):
            raise ValueError(
                f"id_data (type: {type(id_data)}) and id_mapper (type: {type(id_mapper)}) should either both be assigned or both be None."  # noqa
            )
        elif id_data is None and id_mapper is None:
            _, id_data, id_mapper = self.structures_to_arrays(dummy_structures)

        assert (
            id_data is not None and id_mapper is not None
        ), "id_data and id_mapper should not be None at this point"

        # keys = [
        #     ATOM_KEY,
        #     FRAC_KEY,
        #     ABC_KEY,
        #     ANGLES_KEY,
        #     VOLUME_KEY,
        #     SPACE_GROUP_KEY,
        #     DISTANCE_KEY,
        # ]

        [a.shape for a in np.array_split(data, [12], axis=1)]

        zero_pad = 12
        left_arr, right_arr = np.array_split(data, [zero_pad], axis=1)
        _, bottom_left = np.array_split(left_arr, [zero_pad], axis=2)

        verts = np.array_split(bottom_left, np.cumsum([1, 3, 3, 3, 1]), axis=1)

        top_right, bottom_right = np.array_split(right_arr, [zero_pad], axis=2)
        distance_arr = bottom_right

        horzs = np.array_split(top_right, np.cumsum([1, 3, 3, 3, 1]), axis=2)

        def average_vert_horz(vert, horz):
            vert = np.float64(vert)
            horz = np.float64(horz)
            avg = (vert.swapaxes(1, 2) + horz) / 2
            return avg

        avgs = [average_vert_horz(v, h) for v, h in zip(verts, horzs)]

        (
            atom_arr,
            frac_arr,
            abc_arr,
            angles_arr,
            volume_arr,
            space_group_arr,
        ) = avgs

        return (
            atom_arr,
            frac_arr,
            abc_arr,
            angles_arr,
            volume_arr,
            space_group_arr,
            distance_arr,
        )

    def arrays_to_structures(self, data: np.ndarray):
        """Convert scaled crystal (xtal) arrays to pymatgen Structures.

        Parameters
        ----------
        data : np.ndarray
            3D array containing crystallographic information.
        """
        arrays = self.disassemble_blocks(data)

        (
            atom_scaled,
            frac_scaled,
            abc_scaled_tmp,
            angles_scaled_tmp,
            volume_scaled_tmp,
            space_group_scaled_tmp,
            distance_scaled,
        ) = [np.squeeze(arr, axis=2) if arr.shape[2] == 1 else arr for arr in arrays]

        abc_scaled = np.mean(abc_scaled_tmp, axis=1, where=abc_scaled_tmp != 0)
        angles_scaled = np.mean(angles_scaled_tmp, axis=1, where=angles_scaled_tmp != 0)

        volume_scaled = np.mean(volume_scaled_tmp, axis=1)
        space_group_scaled = np.round(np.mean(space_group_scaled_tmp, axis=1)).astype(
            int
        )

        atomic_numbers = rgb_unscaler(atom_scaled, data_range=self.atom_range)
        frac_coords = rgb_unscaler(frac_scaled, data_range=self.frac_range)
        abc = rgb_unscaler(abc_scaled, data_range=self.abc_range)
        angles = rgb_unscaler(angles_scaled, data_range=self.angles_range)

        # # volume, space_group, distance_matrix unecessary for making Structure
        volume = rgb_unscaler(volume_scaled, data_range=self.volume_range)
        space_group = rgb_unscaler(
            space_group_scaled, data_range=self.space_group_range
        )
        distance_matrix = rgb_unscaler(distance_scaled, data_range=self.distance_range)
        # technically unused, but to avoid issue with pre-commit for now:
        volume, space_group, distance_matrix

        atomic_numbers = np.round(atomic_numbers).astype(int)

        # REVIEW: round fractional coordinates to nearest multiple?

        # TODO: tweak lattice parameters to match predicted space group rules

        # build Structure-s
        S: List[Structure] = []
        for i in range(len(atomic_numbers)):
            at = atomic_numbers[i]
            fr = frac_coords[i]
            # di = distance_matrix[i]
            site_ids = np.where(at > 0)

            at = at[site_ids]
            fr = fr[site_ids]
            # di_cropped = di[site_ids[0]][:, site_ids[0]]

            a, b, c = abc[i]
            alpha, beta, gamma = angles[i]

            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
            )
            structure = Structure(lattice, at, fr)
            S.append(structure)

        return S


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
    parser.add_argument(
        dest="fpath",
        help="Crystallographic information file (CIF) filepath (extension must be .cif or .CIF) or path to directory containing .cif files.",  # noqa: E501
        type=str,
        metavar="STRING",
    )
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
    if Path(args.fpath).suffix in [".cif", ".CIF"]:
        fpaths = [args.fpath]
    elif path.isdir(args.fpath):
        fpaths = glob(path.join(args.fpath, "*.cif"))
    else:
        raise ValueError(
            f"Input should be a path to a single .cif file or a path to a directory containing cif file(s). Received: {args.fpath}"  # noqa: E501
        )

    XtalConverter().xtal2png(fpaths)
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

# def mask_png(self, keep_keys:List[str]=[ATOM_KEY]):
#     self.data[self.id_data == self.id_mapper[ATOM_KEY]]

# id_mapper = dict(
#     atom=ATOM_ID,
#     frac=FRAC_ID,
#     abc=ABC_ID,
#     angles=ANGLES_ID,
#     volume=VOLUME_ID,
#     space_group=SPACE_GROUP_ID,
#     distance=DISTANCE_ID,
# )


# (
#     atom_ids,
#     frac_ids,
#     abc_ids,
#     angles_ids,
#     volume_ids,
#     space_group_ids,
#     distance_ids,
# )

# [
#     data[
#         min(i[0]) : max(i[0]) + 1,
#     ]
#     for i in ids
# ]

# (
#     atom_vert,
#     frac_vert,
#     abc_vert,
#     angles_vert,
#     volume_vert,
#     space_group_vert,
#     distance_vert,
# )

# (
#     atom_horz,
#     frac_horz,
#     abc_horz,
#     angles_horz,
#     volume_horz,
#     space_group_horz,
#     distance_horz,
# )

# atom_arr = (atom_vert.swapaxes(1, 2) + atom_horz) / 2
# frac_arr = (frac_vert.swapaxes(1, 2) + frac_horz) / 2
# abc_arr = (abc_vert.swapaxes(1, 2) + abc_horz) / 2

# ids = [np.where(id_data == id_mapper[key]) for key in keys]

# orig_shape = data.shape
# data.flatten()[np.where(id_data.flatten() == 1)[0]]

# atom_scaled = np.array(atomic_numbers)
# frac_scaled = frac_coords
# abc_scaled = np.array(abc)
# angles_scaled = np.array(angles)
# volume_scaled = np.array(volume)
# space_group_scaled = np.array(space_group)
# distance_scaled = distance_matrix

# def count_zero(X):
#     num_zeros = np.count_nonzero(at == 0)
#     return num_zeros
# num_zeros = np.count_nonzero(at == 0)


# atomic_numbers = []
# frac_coords = []
# abc = []
# angles = []
# volume = []
# space_group = []
# distance_matrix = []

# atomic_numbers.append(at)
# frac_coords.append(fr)
# abc.append(abc_tmp[i])
# angles.append(angles_tmp[i])
# volume.append(volume_tmp[i])
# space_group.append(space_group_tmp[i])
# distance_matrix.append(di_cropped)
