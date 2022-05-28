from os import path

from xtal2png.core import main, parse_args


def test_parse_encode_args():
    args = parse_args(["--encode", "--path", "src/xtal2png/utils/Zn2B2PbO6.cif"])
    return args


def test_parse_decode_args():
    args = parse_args(
        ["--decode", "--path", "data/external/preprocessed/Zn2B2PbO6.png"]
    )
    return args


def test_encode_single():
    fpath = path.join("src", "xtal2png", "utils", "Zn2B2PbO6.cif")
    args = ["--encode", "--path", fpath, "--save-dir", "tmp"]
    main(args)


def test_encode_dir():
    fpath = path.join("src", "xtal2png", "utils")
    args = ["--encode", "--path", fpath, "--save-dir", path.join("tmp", "tmp")]
    main(args)


def test_decode_single():
    fpath = path.join("data", "preprocessed", "Zn8B8Pb4O24,volume=623,uid=b62a.png")
    args = ["--decode", "--path", fpath, "--save-dir", "tmp"]
    main(args)


def test_decode_dir():
    fpath = path.join("data", "preprocessed")
    args = ["--decode", "--path", fpath, "--save-dir", "tmp"]
    main(args)


if __name__ == "__main__":
    args = test_parse_encode_args()
    args = test_parse_decode_args()
    test_encode_single()
    test_encode_dir()
    test_decode_single()
    test_decode_dir()

    1 + 1
