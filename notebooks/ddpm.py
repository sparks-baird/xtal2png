from os import path
from pathlib import Path
from uuid import uuid4

from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet
from mp_time_split.core import MPTimeSplit

from xtal2png.core import XtalConverter

mpt = MPTimeSplit()
mpt.load()

fold = 0
train_inputs, val_inputs, train_outputs, val_outputs = mpt.get_train_and_val_data(fold)

data_path = path.join("data", "preprocessed", "mp-time-split", f"fold={fold}")
xc = XtalConverter(save_dir=data_path)
xc.xtal2png(train_inputs.tolist())

model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1).cuda()

diffusion = GaussianDiffusion(
    model, channels=1, image_size=64, timesteps=1000, loss_type="l1"
).cuda()

train_batch_size = 32
print("train_batch_size: ", train_batch_size)

results_folder = path.join("data", "interim", "ddpm", f"fold={fold}", str(uuid4())[0:4])
Path(results_folder).mkdir(exist_ok=True, parents=True)

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

trainer.train()

sampled_images = diffusion.sample(batch_size=100)

1 + 1
