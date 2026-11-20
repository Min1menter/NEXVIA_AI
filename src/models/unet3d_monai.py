from monai.networks.layers import Norm
from monai.networks.nets import UNet


def build_unet() -> UNet:
    """3D (residual) U-Net from MONAI: background + foreground (spleen)."""
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm=Norm.BATCH,
    )
