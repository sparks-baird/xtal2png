import click
from xtal2png import __version__
from .core import setup_logging
import logging
from .core import _logger
from .core import XtalConverter
from glob import glob


@click.command("cli")
@click.option("--version", is_flag=True, help="Show version.")
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    help="Crystallographic information file (CIF) filepath (extension must be .cif or .CIF) or path to directory containing .cif files or processed PNG filepath or path to directory containing processed .png files (extension must be .png or .PNG). Assumes CIFs if --encode flag is used. Assumes PNGs if --decode flag is used.",
    dir_okay=True,
    file_okay=False,
    readable=True,
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
@click.option("--verbose", "-v", is_flag=True, help="Set loglevel to INFO.")
@click.option("--very-verbose", "-vv", is_flag=True, help="Set loglevel to INFO.")
def cli(version, path, save_dir, runtype, verbose, very_verbose):
    """
    xtal2png command line interface.
    """
    if version:
        click.echo("xtal2png version: {}".format(__version__))

    if verbose:
        setup_logging(loglevel=logging.INFO)
    elif very_verbose:
        setup_logging(loglevel=logging.DEBUG)

    _logger.debug("Beginning conversion to PNG format")

    if runtype == "encode":
        if path is None or len(glob(path / "*.cif")) == 0:
            click.echo(
                "Please specify a path to a CIF file or directory containing CIF files."
            )
            return

        if save_dir is None:
            click.echo("Please specify a path to a directory to save the PNG files.")
            return

        xc = XtalConverter(save_dir=save_dir)
        xc.xtal2png(path, save=True)
    elif runtype == "decode":
        if path is None or len(glob(path / "*.png")) == 0:
            click.echo(
                "Please specify a path to a PNG file or directory containing PNG files."
            )
            return
        if save_dir is None:
            click.echo("Please specify a path to a directory to save the CIF files.")
            return

        xc = XtalConverter(save_dir=save_dir)
        xc.png2xtal(path, save=True)


if __name__ == "__main__":
    cli()
