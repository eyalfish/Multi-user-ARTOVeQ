import torch
import torch.nn as nn


# Since the decoder is a vision transformer we will have the encoder as the first few blocks
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsampling module.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Downsampled feature map.
        """
        z = self.conv1_block(x)  # Downsampling: Halves the spatial dimensions
        z = self.conv2_block(z)  # Keeps dimensions the same
        return z


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        self.deconv1_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.deconv2_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the upsampling module.
        :param x: Latent feature map (downsampled)
        :return: Reconstructed image.
        """
        z = self.deconv1_block(x)  # Keeps spatial dimensions
        z = self.deconv2_block(z)  # Upsamples to original size
        return z

class Autoencoder(nn.Module):


    def __init__(self, in_channels, out_channels):
        """ Combines DownSample (encoder) and UpSample (decoder). """
        super(Autoencoder, self).__init__()

        self.downsample = DownSample(in_channels, out_channels)
        self.upsample = UpSample(out_channels, in_channels)

        # custom weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of the layers in the encoder using Xavier initialization
        """
        # Iterate over all the modules in the encoder
        for module in self.modules():
            # Check if the module has a 'weight' attribute (e.g., Conv2d layers)
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)

            # Optionally, initialize biases if they exist
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the encoder
        :param x (torch.Tensor): Input image of size BxCXHXW
        :return: flattened latent state representation
        """
        latent = self.downsample(x) # B, out_channels, H, W
        x_hat = self.upsample(latent) # reconstructed image
        # return self.fully_connected(z.flatten())
        return x_hat
