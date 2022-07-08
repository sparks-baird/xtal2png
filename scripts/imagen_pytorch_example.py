from os import path
from uuid import uuid4

import torch
from imagen_pytorch import BaseUnet64, Imagen, ImagenTrainer, Unet
from mp_time_split.core import MPTimeSplit

from xtal2png.core import XtalConverter

low_mem = False
max_batch_size = 16

mpt = MPTimeSplit()
mpt.load()

fold = 0
train_inputs, val_inputs, train_outputs, val_outputs = mpt.get_train_and_val_data(fold)

xc = XtalConverter(save_dir="tmp", encode_as_primitive=True, decode_as_primitive=True)
arrays, _, _ = xc.structures_to_arrays(train_inputs.tolist(), rgb_scaling=False)
training_images = torch.from_numpy(arrays).float().cuda()

# unets for unconditional imagen

unet1 = Unet(
    dim=32,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=3,
    layer_attns=(False, True, True),
    layer_cross_attns=(False, True, True),
    use_linear_attn=True,
)

if low_mem:
    unet2 = Unet(
        dim=32,
        dim_mults=(1, 2, 4),
        num_resnet_blocks=3,
        layer_attns=(False, True, True),
        layer_cross_attns=(False, True, True),
        use_linear_attn=True,
    )
else:
    # unet2 = SRUnet256()
    unet2 = BaseUnet64()

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    condition_on_text=False,  # this must be set to False for unconditional Imagen
    unets=(unet1, unet2),
    channels=1,
    image_sizes=(32, 64),
    timesteps=1000,
)

trainer = ImagenTrainer(imagen).cuda()

# train each unet in concert, or separately (recommended) to completion

for u in (1, 2):
    loss = trainer(training_images, unet_number=u, max_batch_size=max_batch_size)
    trainer.update(unet_number=u)

# do the above for many many many many steps
# now you can sample images unconditionally from the cascading unet(s)

images = trainer.sample(batch_size=16, return_pil_images=True)  # (16, 3, 128, 128)

results_folder = path.join(
    "data", "interim", "imagen-pytorch", f"fold={fold}", str(uuid4())[0:4] + ".pt"
)
trainer.save(results_folder)
1 + 1
