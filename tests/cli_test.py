from os import path

from click.testing import CliRunner

from xtal2png.cli import cli


def test_encode_single():
    fpath = path.abspath(path.join("src", "xtal2png", "utils", "Zn2B2PbO6.cif"))

    args = ["--encode", "--path", fpath, "--save-dir", "tmp"]

    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0


def test_encode_dir():
    fpath = path.abspath(path.join("src", "xtal2png", "utils"))
    args = ["--encode", "--path", fpath, "--save-dir", path.join("tmp", "tmp")]
    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0


def test_decode_single():
    fpath = path.join(
        "data", "preprocessed", "examples", "Zn8B8Pb4O24,volume=623,uid=b62a.png"
    )
    args = ["--decode", "--path", fpath, "--save-dir", "tmp"]
    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0


def test_decode_dir():
    fpath = path.join("data", "preprocessed", "examples")
    args = ["--decode", "--path", fpath, "--save-dir", "tmp"]
    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert result.exit_code == 0


if __name__ == "__main__":
    test_encode_single()
    test_encode_dir()
    test_decode_single()
    test_decode_dir()
