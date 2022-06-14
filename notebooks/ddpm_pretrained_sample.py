import os
import re
from os import path
from pathlib import Path

import numpy as np
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
from mp_time_split.core import MPTimeSplit
from PIL import Image
from pymatgen.core.composition import Composition
from pymatviz.elements import ptable_heatmap_plotly

from xtal2png.core import XtalConverter
from xtal2png.utils.data import rgb_scaler

mpt = MPTimeSplit()
mpt.load()

fold = 0
train_inputs, val_inputs, train_outputs, val_outputs = mpt.get_train_and_val_data(fold)

model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1).cuda()

diffusion = GaussianDiffusion(
    model, channels=1, image_size=64, timesteps=1000, loss_type="l1"
).cuda()

train_batch_size = 32
print("train_batch_size: ", train_batch_size)

uid = "427c"
results_folder = path.join("data", "interim", "ddpm", f"fold={fold}", uid)
Path(results_folder).mkdir(exist_ok=True, parents=True)

data_path = path.join("data", "preprocessed", "mp-time-split", f"fold={fold}")

fnames = os.listdir(results_folder)

# i.e. "model-1.pt" --> "1.pt" --> "1" --> 1
checkpoints = [int(name.split("-")[1].split(".")[0]) for name in fnames]
checkpoint = np.max(checkpoints)


trainer = Trainer(
    diffusion,
    data_path,
    image_size=64,
    train_batch_size=train_batch_size,
    train_lr=2e-5,
    train_num_steps=700000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    augment_horizontal_flip=False,
    results_folder=results_folder,
)

trainer.load(checkpoint)

diffusion = trainer.model

img_arrays_torch = diffusion.sample(batch_size=16)
unscaled_arrays = np.squeeze(img_arrays_torch.cpu().numpy())
rgb_arrays = rgb_scaler(unscaled_arrays, data_range=(0, 1))

sampled_images = [Image.fromarray(arr, "I") for arr in rgb_arrays]

gen_path = path.join(
    "data", "preprocessed", "mp-time-split", "ddpm", f"fold={fold}", uid
)
xc = XtalConverter(save_dir=gen_path)
structures = xc.png2xtal(sampled_images, save=True)

space_group = []
W = []
for s in structures:
    try:
        space_group.append(s.get_space_group_info(symprec=0.1)[1])
    except Exception as e:
        W.append(e)
        space_group.append(None)
print(space_group)

equimolar_compositions = train_inputs.apply(
    lambda s: Composition(re.sub(r"\d", "", s.formula))
)
fig = ptable_heatmap_plotly(equimolar_compositions)
fig.show()

1 + 1

# %% Code Graveyard
# compositions = train_inputs.apply(lambda s: s.composition)
# atomic_numbers = train_inputs.apply(lambda s: np.unique(s.atomic_numbers))
