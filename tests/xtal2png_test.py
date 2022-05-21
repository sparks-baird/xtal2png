# convert between a fixed crystal Structure to numeric representation (test), to PNG
# (test), and back to crystal Structure (test)

from os import path

import pandas as pd
from pymatgen.core.structure import Structure

from xtal2png.skeleton import XtalConverter as xc

EXAMPLE_CIFS = ["Zn2B2PbO6.cif", "V2NiSe4.cif"]
fpaths = [path.join("data", "external", cif) for cif in EXAMPLE_CIFS]
S = [Structure.from_file(fpath) for fpath in fpaths]

data = xc.structures_to_arrays(S)

for i, fpath in enumerate(fpaths):
    savepath = path.splitext(fpath)[0] + ".csv"
    pd.DataFrame(data[i]).to_csv(savepath)

xc.structures_to_arrays([S[0]])

1 + 1
