from os import path
from pathlib import Path
from uuid import uuid4

import torch
from imagen_pytorch import Imagen, ImagenTrainer, Unet
from mp_time_split.core import MPTimeSplit

from xtal2png.core import XtalConverter

mpt = MPTimeSplit()
mpt.load()

fold = 0
train_inputs, val_inputs, train_outputs, val_outputs = mpt.get_train_and_val_data(fold)

data_path = path.join("data", "preprocessed", "mp-time-split", f"fold={fold}")
xc = XtalConverter(
    save_dir=data_path, encode_as_primitive=True, decode_as_primitive=True
)
xc.xtal2png(train_inputs.tolist())

max_batch_size = 32

results_folder = path.join("data", "interim", "ddpm", f"fold={fold}", str(uuid4())[0:4])
Path(results_folder).mkdir(exist_ok=True, parents=True)

# unets for unconditional imagen

unet1 = Unet(
    dim=32,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=3,
    layer_attns=(False, True, True),
    layer_cross_attns=(False, True, True),
    use_linear_attn=True,
)

unet2 = Unet(
    dim=32,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=3,
    layer_attns=(False, True, True),
    layer_cross_attns=(False, True, True),
    use_linear_attn=True,
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    condition_on_text=False,  # this must be set to False for unconditional Imagen
    unets=(unet1, unet2),
    channels=1,
    image_sizes=(16, 32),
    timesteps=1000,
)

trainer = ImagenTrainer(imagen).cuda()

# now get a ton of images and feed it through the Imagen trainer

training_images = torch.randn(4, 1, 64, 64).cuda()

# train each unet in concert, or separately (recommended) to completion

for u in (1, 2):
    loss = trainer(training_images, unet_number=u, max_batch_size=max_batch_size)
    trainer.update(unet_number=u)

# do the above for many many many many steps
# now you can sample images unconditionally from the cascading unet(s)

images = trainer.sample(batch_size=16)  # (16, 3, 128, 128)

1 + 1
