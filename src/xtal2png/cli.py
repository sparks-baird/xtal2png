import logging
import os
from glob import glob

import click
from click._compat import get_text_stderr
from click.exceptions import UsageError
from click.utils import echo

from xtal2png import __version__
from xtal2png.core import XtalConverter, _logger, setup_logging


def _show_usage_error(self, file=None):
    if file is None:
        file = get_text_stderr()
    color = None

    echo("Error: %s" % self.format_message(), file=file, color=color)
    if self.ctx is not None:
        color = self.ctx.color
        echo("\n\n" + self.ctx.get_help() + "\n", file=file, color=color)


UsageError.show = _show_usage_error  # type: ignore


def check_save_dir(save_dir):
    if save_dir is None:
        raise UsageError("Please specify a path to a directory to save the PNG files.")


def check_path(path, extension):
    if path is None:
        raise UsageError(
            f"Please specify a path to a {extension} file or "
            f"directory containing {extension} files."
        )


def check_files(path, extension):
    if os.path.isdir(path):
        files = glob(os.path.join(path, f"*.{extension}"))
        if not files:
            raise UsageError(f"No {extension.upper()} files found in directory: {path}")
    elif os.path.isfile(path):
        if not path.endswith(f".{extension}"):
            raise UsageError(f"File must have .{extension} extension: {path}")
        files = [path]

    return files


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
@click.option(
    "--max-sites",
    "-ms",
    type=int,
    default=52,
    help="Maximum number of sites to accomodate in encoding, by default 52",
)
@click.pass_context
def cli(ctx, version, path, save_dir, runtype, verbose, very_verbose, max_sites):
    """
    xtal2png command line interface.
    """
    if version:
        click.echo("xtal2png version: {}".format(__version__))
        return
    if verbose:
        setup_logging(loglevel=logging.INFO)
    if very_verbose:
        setup_logging(loglevel=logging.DEBUG)

    if not runtype and (path or save_dir):
        raise UsageError("Please specify --encode or --decode.")

    _logger.debug("Beginning conversion to PNG format")

    if runtype == "encode":
        check_path(path, "CIF")
        check_save_dir(save_dir)

        files = check_files(path, "cif")

        xc = XtalConverter(save_dir=save_dir, max_sites=max_sites)
        xc.xtal2png(files, save=True)
        return

    elif runtype == "decode":
        check_path(path, "PNG")
        check_save_dir(save_dir)

        files = check_files(path, "png")

        xc = XtalConverter(save_dir=save_dir, max_sites=max_sites)
        xc.png2xtal(files, save=True)
        return

    click.echo(ctx.get_help())


if __name__ == "__main__":
    cli()
