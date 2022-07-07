from cProfile import run
import logging
import os
from glob import glob

import click

from xtal2png import __version__

from .core import XtalConverter, _logger, setup_logging

from click.exceptions import UsageError
from click._compat import get_text_stderr
from click.utils import echo


def _show_usage_error(self, file=None):
    if file is None:
        file = get_text_stderr()
    color = None

    echo("Error: %s" % self.format_message(), file=file, color=color)
    if self.ctx is not None:
        color = self.ctx.color
        echo("\n\n" + self.ctx.get_help() + "\n", file=file, color=color)


UsageError.show = _show_usage_error


@click.command("cli")
@click.option("--version", is_flag=True, help="Show version.")
@click.option(
    "--path",
    "-p",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=True,
        readable=True,
    ),
    help="Crystallographic information file (CIF) filepath "
    " (extension must be .cif or .CIF)"
    " or path to directory containing .cif files or processed PNG filepath"
    " or path to directory containing processed .png files "
    "(extension must be .png or .PNG). "
    "Assumes CIFs if --encode flag is used. Assumes PNGs if --decode flag is used.",
)
@click.option(
    "--save-dir",
    "-s",
    type=click.Path(exists=False),
    help="Encode CIF files as PNG images.",
)
@click.option(
    "--encode", "runtype", flag_value="encode", help="Encode CIF files as PNG images."
)
@click.option(
    "--decode", "runtype", flag_value="decode", help="Decode PNG images to CIF files."
)
@click.option("--verbose", "-v", help="Set loglevel to INFO.")
@click.option("--very-verbose", "-vv", help="Set loglevel to INFO.")
def cli(version, path, save_dir, runtype, verbose, very_verbose):
    """
    xtal2png command line interface.
    """
    if version:
        click.echo("xtal2png version: {}".format(__version__))
        return
    if verbose:
        setup_logging(loglevel=logging.INFO)
    elif very_verbose:
        setup_logging(loglevel=logging.DEBUG)

    if not runtype:
        raise UsageError("Please specify --encode or --decode.")

    _logger.debug("Beginning conversion to PNG format")
    if runtype == "encode":
        if (
            path is None
            or (
                len(glob(os.path.join(path, "*.cif")))
                if os.path.isdir(path)
                else path.endswith("cif")
            )
            == 0
        ):
            raise UsageError(
                "Please specify a path to a CIF file or directory containing CIF files."
            )

        if save_dir is None:
            raise UsageError(
                "Please specify a path to a directory to save the PNG files."
            )
            return

        xc = XtalConverter(save_dir=save_dir)
        xc.xtal2png(path, save=True)
    elif runtype == "decode":
        if path is None or (
            len(glob(os.path.join(path, "*.png")))
            if os.path.isdir(path)
            else path.endswith("png")
        ):
            raise UsageError(
                "Please specify a path to a PNG file or directory containing PNG files."
            )
            return
        if save_dir is None:
            raise UsageError(
                "Please specify a path to a directory to save the CIF files."
            )
            return

        xc = XtalConverter(save_dir=save_dir)
        xc.png2xtal(path, save=True)


if __name__ == "__main__":
    cli()
