# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2DAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=1024):
        super(UNet2DAutoencoder, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = self.conv_block(input_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Flatten and fully connected layer for encoding
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 32 * 32, latent_dim)  # Updated

        # Decoder fully connected layer
        self.fc2 = nn.Linear(latent_dim, 256 * 32 * 32)  # Updated
        self.unflatten = nn.Unflatten(1, (256, 32, 32))  # Updated

        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 32)

        # Final layer
        self.final = nn.Conv2d(32, input_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def encode(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))  # 256x256
        e3 = self.enc3(F.max_pool2d(e2, 2))  # 128x128
        e4 = self.enc4(F.max_pool2d(e3, 2))  # 64x64

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))  # 32x32

        # Flatten and pass through fc1
        b_flat = self.flatten(b)  # [B, 256*32*32] = [B, 262144]
        latent_vector = self.fc1(b_flat)  # [B, latent_dim]

        return latent_vector

    def decode(self, latent_vector):
        # Pass through fc2 and reshape
        x = self.fc2(latent_vector)  # [B, 256*32*32] = [B, 262144]
        x = self.unflatten(x)        # [B, 256, 32, 32]

        # Decoder
        d4 = self.up4(x)  # [B, 256, 64, 64]
        d4 = self.dec4(d4)  # [B, 256, 64, 64]

        d3 = self.up3(d4)  # [B, 128, 128, 128]
        d3 = self.dec3(d3)  # [B, 128, 128, 128]

        d2 = self.up2(d3)  # [B, 64, 256, 256]
        d2 = self.dec2(d2)  # [B, 64, 256, 256]

        d1 = self.up1(d2)  # [B, 32, 512, 512]
        d1 = self.dec1(d1)  # [B, 32, 512, 512]

        # Final upsampling to match original size
        d0 = F.interpolate(d1, scale_factor=1, mode='bilinear', align_corners=True)  # No scaling needed
        # Optionally, add another conv_block here if desired
        # d0 = self.conv_block(32, 32)(d0)  # Uncomment if needed

        # Final output
        out = self.final(d0)  # [B, 1, 512, 512]
        return torch.sigmoid(out)

    def forward(self, x):
        latent_vector = self.encode(x)
        reconstruction = self.decode(latent_vector)
        return reconstruction

    def get_latent_vector(self, x):
        latent_vector = self.encode(x)
        return latent_vector


# class UNet2DAutoencoder(nn.Module):
#     def __init__(self, input_channels=1, latent_dim=1):
#         super(UNet2DAutoencoder, self).__init__()

#         self.latent_dim = latent_dim

#         # Encoder
#         self.enc1 = self.conv_block(input_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.enc4 = self.conv_block(128, 256)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True)
#         )

#         # Flatten and fully connected layer for encoding
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 8 * 8, latent_dim)

#         # Decoder fully connected layer
#         self.fc2 = nn.Linear(latent_dim, 256 * 8 * 8)
#         self.unflatten = nn.Unflatten(1, (256, 8, 8))

#         # Decoder
#         self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.dec4 = self.conv_block(256, 256)

#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = self.conv_block(128, 128)

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = self.conv_block(64, 64)

#         self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = self.conv_block(32, 32)

#         # Final layer
#         self.final = nn.Conv2d(32, input_channels, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#     def encode(self, x):
#         # Encoder
#         e1 = self.enc1(x)
#         e2 = self.enc2(F.max_pool2d(e1, 2))
#         e3 = self.enc3(F.max_pool2d(e2, 2))
#         e4 = self.enc4(F.max_pool2d(e3, 2))

#         # Bottleneck
#         b = self.bottleneck(F.max_pool2d(e4, 2))  # [B, 256, 8, 8]

#         # Flatten and pass through fc1
#         b_flat = self.flatten(b)  # [B, 256*8*8] = [B, 16384]
#         latent_vector = self.fc1(b_flat)  # [B, latent_dim]

#         return latent_vector

#     def decode(self, latent_vector):
#         # Pass through fc2 and reshape
#         x = self.fc2(latent_vector)  # [B, 16384]
#         x = self.unflatten(x)        # [B, 256, 8, 8]

#         # Decoder
#         d4 = self.up4(x)  # [B, 256, 16, 16]
#         d4 = self.dec4(d4)  # [B, 256, 16, 16]

#         d3 = self.up3(d4)  # [B, 128, 32, 32]
#         d3 = self.dec3(d3)  # [B, 128, 32, 32]

#         d2 = self.up2(d3)  # [B, 64, 64, 64]
#         d2 = self.dec2(d2)  # [B, 64, 64, 64]

#         d1 = self.up1(d2)  # [B, 32, 128, 128]
#         d1 = self.dec1(d1)  # [B, 32, 128, 128]

#         # Final output
#         out = self.final(d1)  # [B, 1, 128, 128]
#         return torch.sigmoid(out)

#     def forward(self, x):
#         latent_vector = self.encode(x)
#         reconstruction = self.decode(latent_vector)
#         return reconstruction


#     def get_latent_vector(self, x):
#         """
#         Extracts the latent vector from the input.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [B, 1, H, W].
        
#         Returns:
#             torch.Tensor: Latent vector of shape [B, latent_dim].
#         """
#         latent_vector = self.encode(x)
#         return latent_vector

# class UNet3DAutoencoder(nn.Module):
#     def __init__(self, input_channels=1, latent_dim=512):
#         super(UNet3DAutoencoder, self).__init__()

#         self.latent_dim = latent_dim

#         # Encoder
#         self.enc1 = self.conv_block(input_channels, 32)
#         self.enc2 = self.conv_block(32, 64)
#         self.enc3 = self.conv_block(64, 128)
#         self.enc4 = self.conv_block(128, 256)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv3d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm3d(512),
#             nn.ReLU(True),
#             nn.Conv3d(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm3d(256),
#             nn.ReLU(True)
#         )

#         # Flatten and fully connected layer for encoding
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 4 * 4 * 4, latent_dim)  # 16384 -> latent_dim

#         # Decoder fully connected layer
#         self.fc2 = nn.Linear(latent_dim, 256 * 4 * 4 * 4)  # latent_dim -> 16384
#         self.unflatten = nn.Unflatten(1, (256, 4, 4, 4))

#         # Decoder
#         self.up4 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
#         self.dec4 = self.conv_block(256 + 256, 256)

#         self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
#         self.dec3 = self.conv_block(128 + 128, 128)

#         self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
#         self.dec2 = self.conv_block(64 + 64, 64)

#         self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
#         self.dec1 = self.conv_block(32 + 32, 32)

#         # Final layer
#         self.final = nn.Conv3d(32, input_channels, kernel_size=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(True)
#         )

#     def encode(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(F.max_pool3d(e1, 2))
#         e3 = self.enc3(F.max_pool3d(e2, 2))
#         e4 = self.enc4(F.max_pool3d(e3, 2))

#         b = self.bottleneck(F.max_pool3d(e4, 2))

#         b_flat = self.flatten(b)
#         latent_vector = self.fc1(b_flat)

#         return latent_vector, (e1, e2, e3, e4)

#     def decode(self, latent_vector, enc_features):
#         x = self.fc2(latent_vector)  
#         x = self.unflatten(x) 
#         e1, e2, e3, e4 = enc_features

#         d4 = self.up4(x)
#         d4 = torch.cat([d4, e4], dim=1)
#         d4 = self.dec4(d4)

#         d3 = self.up3(d4)
#         d3 = torch.cat([d3, e3], dim=1)
#         d3 = self.dec3(d3)

#         d2 = self.up2(d3)
#         d2 = torch.cat([d2, e2], dim=1)
#         d2 = self.dec2(d2)

#         d1 = self.up1(d2)
#         d1 = torch.cat([d1, e1], dim=1)
#         d1 = self.dec1(d1)

#         out = self.final(d1)
#         return torch.sigmoid(out)

#     def forward(self, x):
#         latent_vector, enc_features = self.encode(x)
#         reconstruction = self.decode(latent_vector, enc_features)
#         return reconstruction

#     def get_latent_vector(self, x):
#         """
#         Extracts the latent vector from the input.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [B, 1, D, H, W].
        
#         Returns:
#             torch.Tensor: Latent vector of shape [B, latent_dim].
#         """
#         latent_vector, _ = self.encode(x)
#         return latent_vector

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        self.mlp = MLP(input_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input if needed
        x.requires_grad_(True)  # Ensure gradients are tracked
        output = self.mlp(x)
        return output, x  # Return the output and the inputs for the custom loss function

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, output_size)

        self.dropout = nn.Dropout(0.5)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x
