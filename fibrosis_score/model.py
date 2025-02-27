import torch
from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
import torch.nn as nn

# downsample 
class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor: int):
        """Initialize a MaxPool2d module with the input downsample fator"""

        super().__init__()

        self.downsample_factor = downsample_factor
        # SOLUTION 2B1: Initialize the maxpool module
        self.down = torch.nn.MaxPool2d(downsample_factor)

    def check_valid(self, image_size: tuple[int, int]) -> bool:
        """Check if the downsample factor evenly divides each image dimension.
        Returns `True` for valid image sizes and `False` for invalid image sizes.
        Note: there are multiple ways to do this!
        """
        # SOLUTION 2B2: Check that the image_size is valid to use with the downsample factor
        for dim in image_size:
            if dim % self.downsample_factor != 0:
                return False
        return True

    def forward(self, x):
        if not self.check_valid(tuple(x.size()[2:])):
            raise RuntimeError(
                "Can not downsample shape %s with factor %s"
                % (x.size(), self.downsample_factor)
            )

        return self.down(x)
    
# Convolution Block 
class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
    ):
        """A convolution block for a U-Net. Contains two convolutions, each followed by a ReLU.

        Args:
            in_channels (int): The number of input channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            out_channels (int): The number of output channels for this conv block. Depends on
                the layer and side of the U-Net and the hyperparameters.
            kernel_size (int): The size of the kernel. A kernel size of N signifies an
                NxN square kernel.
            padding (str): The type of convolution padding to use. Either "same" or "valid".
                Defaults to "same".
        """
        super().__init__()

        if kernel_size % 2 == 0:
            msg = "Only allowing odd kernel sizes."
            raise ValueError(msg)

        # SOLUTION 3.1: Initialize your modules and define layers.
        self.conv_pass = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        # SOLUTION 3.2: Apply the modules you defined to the input x
        return self.conv_pass(x)
    
# crop
def center_crop(x, y):
    """Center-crop x to match spatial dimensions given by y."""

    x_target_size = x.size()[:2] + y.size()[2:]

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]

# crop and concatenate
class CropAndConcat(torch.nn.Module):
    def forward(self, encoder_output, upsample_output):
        # SOLUTION 4: Implement the forward function
        encoder_cropped = center_crop(encoder_output, upsample_output)

        return torch.cat([encoder_cropped, upsample_output], dim=1)
    
# Output Convolution
class OutputConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module | None = None,
    ):
        """
        A module that uses a convolution with kernel size 1 to get the appropriate
        number of output channels, and then optionally applies a final activation.

        Args:
            in_channels (int): The number of feature maps that will be input to the
                OutputConv block.
            out_channels (int): The number of channels that you want in the output
            activation (str | None, optional): Accepts the name of any torch activation
                function  (e.g., ``ReLU`` for ``torch.nn.ReLU``) or None for no final
                activation. Defaults to None.
        """
        super().__init__()

        # SOLUTION 5.1: Define the convolution submodule
        self.final_conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.activation = activation

    def forward(self, x):
        # SOLUTION 5.2: Implement the forward function
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
# Unet Architecture 
class UNet(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int = 1,
        final_activation: torch.nn.Module | None = None,
        num_fmaps: int = 64,
        fmap_inc_factor: int = 2,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        padding: str = "same",
        upsample_mode: str = "nearest",
    ):
        """A U-Net for 2D input that expects tensors shaped like::
            ``(batch, channels, height, width)``.
        Args:
            depth:
                The number of levels in the U-Net. 2 is the smallest that really
                makes sense for the U-Net architecture, as a one layer U-Net is
                basically just 2 conv blocks.
            in_channels:
                The number of input channels in your dataset.
            out_channels (optional):
                How many output channels you want. Depends on your task. Defaults to 1.
            final_activation (optional):
                What activation to use in your final output block. Depends on your task.
                Defaults to None.
            num_fmaps (optional):
                The number of feature maps in the first layer. Defaults to 64.
            fmap_inc_factor (optional):
                By how much to multiply the number of feature maps between
                layers. Encoder layer ``l`` will have ``num_fmaps*fmap_inc_factor**l``
                output feature maps. Defaults to 2.
            downsample_factor (optional):
                Factor to use for down- and up-sampling the feature maps between layers.
                Defaults to 2.
            kernel_size (optional):
                Kernel size to use in convolutions on both sides of the UNet.
                Defaults to 3.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid'. Defaults to "same."
            upsample_mode (optional):
                The upsampling mode to pass to torch.nn.Upsample. Usually "nearest"
                or "bilinear." Defaults to "nearest."
        """

        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_activation = final_activation
        self.num_fmaps = num_fmaps
        self.fmap_inc_factor = fmap_inc_factor
        self.downsample_factor = downsample_factor
        self.kernel_size = kernel_size
        self.padding = padding
        self.upsample_mode = upsample_mode

        # left convolutional passes
        self.left_convs = torch.nn.ModuleList()
        # SOLUTION 6.2A: Initialize list here
        for level in range(self.depth):
            fmaps_in, fmaps_out = self.compute_fmaps_encoder(level)
            self.left_convs.append(
                ConvBlock(fmaps_in, fmaps_out, self.kernel_size, self.padding)
            )

        # right convolutional passes
        self.right_convs = torch.nn.ModuleList()
        # SOLUTION 6.2B: Initialize list here
        for level in range(self.depth - 1):
            fmaps_in, fmaps_out = self.compute_fmaps_decoder(level)
            self.right_convs.append(
                ConvBlock(
                    fmaps_in,
                    fmaps_out,
                    self.kernel_size,
                    self.padding,
                )
            )

        # SOLUTION 6.3: Initialize other modules here
        self.downsample = Downsample(self.downsample_factor)
        self.upsample = torch.nn.Upsample(
            scale_factor=self.downsample_factor,
            mode=self.upsample_mode,
        )
        self.crop_and_concat = CropAndConcat()
        self.final_conv = OutputConv(
            self.compute_fmaps_decoder(0)[1], self.out_channels, self.final_activation
        )

    def compute_fmaps_encoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for
        a conv block at a given level of the UNet encoder (left side).

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the encoder convolutional pass in the given level.
        """
        # SOLUTION 6.1A: Implement this function
        if level == 0:
            fmaps_in = self.in_channels
        else:
            fmaps_in = self.num_fmaps * self.fmap_inc_factor ** (level - 1)

        fmaps_out = self.num_fmaps * self.fmap_inc_factor**level
        return fmaps_in, fmaps_out

    def compute_fmaps_decoder(self, level: int) -> tuple[int, int]:
        """Compute the number of input and output feature maps for a conv block
        at a given level of the UNet decoder (right side). Note:
        The bottom layer (depth - 1) is considered an "encoder" conv pass,
        so this function is only valid up to depth - 2.

        Args:
            level (int): The level of the U-Net which we are computing
            the feature maps for. Level 0 is the input level, level 1 is
            the first downsampled layer, and level=depth - 1 is the bottom layer.

        Output (tuple[int, int]): The number of input and output feature maps
            of the decoder convolutional pass in the given level.
        """
        # SOLUTION 6.1B: Implement this function
        fmaps_out = self.num_fmaps * self.fmap_inc_factor ** (level)
        concat_fmaps = self.compute_fmaps_encoder(level)[
            1
        ]  # The channels that come from the skip connection
        fmaps_in = concat_fmaps + self.num_fmaps * self.fmap_inc_factor ** (level + 1)

        return fmaps_in, fmaps_out

    def forward(self, x):
        # left side
        convolution_outputs = []
        layer_input = x
        for i in range(self.depth - 1):
            # SOLUTION 6.4A: Implement encoder here
            conv_out = self.left_convs[i](layer_input)
            convolution_outputs.append(conv_out)
            downsampled = self.downsample(conv_out)
            layer_input = downsampled

        # bottom
        # SOLUTION 6.4B: Implement bottom of U-Net here
        conv_out = self.left_convs[-1](layer_input)
        layer_input = conv_out

        # right
        for i in range(0, self.depth - 1)[::-1]:
            # SOLUTION 6.4C: Implement decoder here
            upsampled = self.upsample(layer_input)
            concat = self.crop_and_concat(convolution_outputs[i], upsampled)
            conv_output = self.right_convs[i](concat)
            layer_input = conv_output

        # SOLUTION 6.4D: Apply the final convolution and return the output
        return self.final_conv(layer_input)
