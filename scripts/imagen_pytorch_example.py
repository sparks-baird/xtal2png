import torch
from imagen_pytorch import Imagen, ImagenTrainer, SRUnet256, Unet

# unets for unconditional imagen

unet1 = Unet(
    dim=32,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=3,
    layer_attns=(False, True, True),
    layer_cross_attns=(False, True, True),
    use_linear_attn=True,
)

unet2 = SRUnet256(
    dim=32,
    dim_mults=(1, 2, 4),
    num_resnet_blocks=(2, 4, 8),
    layer_attns=(False, False, True),
    layer_cross_attns=(False, False, True),
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    condition_on_text=False,  # this must be set to False for unconditional Imagen
    unets=(unet1, unet2),
    image_sizes=(64, 128),
    timesteps=1000,
)

trainer = ImagenTrainer(imagen).cuda()

# now get a ton of images and feed it through the Imagen trainer

training_images = torch.randn(4, 3, 256, 256).cuda()

# train each unet in concert, or separately (recommended) to completion

for u in (1, 2):
    loss = trainer(training_images, unet_number=u)
    trainer.update(unet_number=u)

# do the above for many many many many steps
# now you can sample images unconditionally from the cascading unet(s)

images = trainer.sample(batch_size=16)  # (16, 3, 128, 128)

1 + 1
