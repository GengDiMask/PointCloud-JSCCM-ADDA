"""
TAE_v2: Improved Transmission AutoEncoder for Point Cloud Compression

Key Improvements over TAE.py:
1. Squeeze-and-Excitation (SE) Attention Blocks
2. Group Normalization (batch-size independent)
3. Pre-Activation Residual Blocks (GN → GELU → Conv)
4. GELU Activation (smoother gradients)

Compatible with existing ADDAChannel from TAE.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Import channel models from original TAE
from TAE import AWGNChannel, ADDAChannel, QuantizationLayer, get_loss


# =============================================================================
# Building Blocks
# =============================================================================

class SEBlock3d(nn.Module):
    """
    Squeeze-and-Excitation Block for 3D Convolutions.
    Adaptively recalibrates channel-wise feature responses.
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)


class PreActResBlock(nn.Module):
    """
    Pre-Activation Residual Block with Group Normalization and GELU.
    
    Structure: GN → GELU → Conv → GN → GELU → Conv + SE + Skip
    
    Reference: He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016
    """
    def __init__(self, channels, groups=8, se_reduction=8):
        super(PreActResBlock, self).__init__()
        
        # Pre-activation path
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        # SE attention
        self.se = SEBlock3d(channels, reduction=se_reduction)
        
    def forward(self, x):
        identity = x
        
        # Pre-activation: GN → GELU → Conv
        out = self.gn1(x)
        out = F.gelu(out)
        out = self.conv1(out)
        
        out = self.gn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        
        # SE attention
        out = self.se(out)
        
        # Residual connection
        out = out + identity
        return out


class PreActTransposedResBlock(nn.Module):
    """
    Pre-Activation Residual Block for Decoder (uses ConvTranspose3d).
    """
    def __init__(self, channels, groups=8, se_reduction=8):
        super(PreActTransposedResBlock, self).__init__()
        
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.ConvTranspose3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.ConvTranspose3d(channels, channels, kernel_size=3, padding=1, bias=False)
        
        self.se = SEBlock3d(channels, reduction=se_reduction)
        
    def forward(self, x):
        identity = x
        
        out = self.gn1(x)
        out = F.gelu(out)
        out = self.conv1(out)
        
        out = self.gn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        
        out = self.se(out)
        out = out + identity
        return out


# =============================================================================
# Encoder / Decoder Networks
# =============================================================================

class EncoderNetworkV2(nn.Module):
    """
    Improved Encoder with SE attention and Pre-Activation ResBlocks.
    """
    def __init__(self, task, num_filters, groups=8 ):
        super(EncoderNetworkV2, self).__init__()
        
        if task == 'geometry':
            ch = 1
        elif task == 'color':
            ch = 3
        elif task == 'geometry+color':
            ch = 4
        else:
            ch = 1
            
        # Initial convolution
        self.conv_in = nn.Conv3d(ch, num_filters, kernel_size=9, stride=2, padding=4)
        self.gn_in = nn.GroupNorm(groups, num_filters)
        
        # Residual block 1
        self.res_block1 = PreActResBlock(num_filters, groups=groups)
        
        # Downsample
        self.conv_down = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        self.gn_down = nn.GroupNorm(groups, num_filters)
        
        # Residual block 2
        self.res_block2 = PreActResBlock(num_filters, groups=groups)
        
        # Output convolution (to latent space)
        self.conv_out = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=4, padding=2, bias=False)
        
        # He initialization for output layer
        init.kaiming_uniform_(self.conv_out.weight, nonlinearity='linear')

    def forward(self, x):
        # Initial conv + GN + GELU
        x = self.conv_in(x)
        x = self.gn_in(x)
        x = F.gelu(x)
        
        # Res block 1
        x = self.res_block1(x)
        
        # Downsample + GN + GELU
        x = self.conv_down(x)
        x = self.gn_down(x)
        x = F.gelu(x)
        
        # Res block 2
        x = self.res_block2(x)
        
        # Output (no activation - goes to channel)
        x = self.conv_out(x)
        return x


class DecoderNetworkV2(nn.Module):
    """
    Improved Decoder with SE attention and Pre-Activation ResBlocks.
    """
    def __init__(self, task, num_filters, groups=8):
        super(DecoderNetworkV2, self).__init__()
        
        if task == 'geometry':
            ch = 1
        elif task == 'color':
            ch = 3
        elif task == 'geometry+color':
            ch = 4
        else:
            ch = 1
            
        # Input convolution (from latent space)
        # 2->8: stride=4, kernel=5, padding=2, output_padding=3
        self.conv_in = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=4, padding=2, output_padding=3)
        self.gn_in = nn.GroupNorm(groups, num_filters)
        
        # Residual block 1
        self.res_block1 = PreActTransposedResBlock(num_filters, groups=groups)
        
        # Upsample
        self.conv_up = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.gn_up = nn.GroupNorm(groups, num_filters)
        
        # Residual block 2
        self.res_block2 = PreActTransposedResBlock(num_filters, groups=groups)
        
        # Output convolution
        self.conv_out = nn.ConvTranspose3d(num_filters, ch, kernel_size=9, stride=2, padding=4, output_padding=1)
        
        # Xavier initialization for output layer (before sigmoid)
        init.xavier_uniform_(self.conv_out.weight)

    def forward(self, x):
        # Input conv + GN + GELU
        x = self.conv_in(x)
        x = self.gn_in(x)
        x = F.gelu(x)
        
        # Res block 1
        x = self.res_block1(x)
        
        # Upsample + GN + GELU
        x = self.conv_up(x)
        x = self.gn_up(x)
        x = F.gelu(x)
        
        # Res block 2
        x = self.res_block2(x)
        
        # Output + Sigmoid
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        return x


# =============================================================================
# AutoEncoder V2
# =============================================================================

class AutoEncoderV2(nn.Module):
    """
    Improved AutoEncoder with SE attention and Pre-Activation ResBlocks.
    
    Compatible with existing ADDA channel model.
    """
    def __init__(self, num_filters, task, snr, 
                 enable_adda=True, adda_bits=8, adda_alpha=1.0, adda_beta=1.0,
                 nonlinearity='none', p=3.0, sat=1.0,
                 dnl_sigma=0.0, inl_gamma=0.01,
                 groups=8):
        super(AutoEncoderV2, self).__init__()
        
        self.encoder = EncoderNetworkV2(task, num_filters, groups=groups)
        self.decoder = DecoderNetworkV2(task, num_filters, groups=groups)
        
        # Channel model (reuse from TAE.py)
        if enable_adda:
            self.channel = ADDAChannel(
                snr=snr, 
                bits=adda_bits, 
                alpha=adda_alpha, 
                beta=adda_beta,
                nonlinearity=nonlinearity,
                p=p,
                sat=sat,
                dnl_sigma=dnl_sigma,
                inl_gamma=inl_gamma
            )
        else:
            self.channel = AWGNChannel(snr)
            
        self.enable_adda = enable_adda

    def forward(self, x):
        # Encode
        latent = self.encoder(x)
        
        # Channel (ADDA or AWGN)
        latent_noisy = self.channel(latent)
        
        # Decode
        output = self.decoder(latent_noisy)
        return output


# =============================================================================
# Utility Functions
# =============================================================================

def get_model_v2(num_filters, task, snr, **kwargs):
    """
    Factory function to create AutoEncoderV2.
    """
    return AutoEncoderV2(num_filters=num_filters, task=task, snr=snr, **kwargs)


def count_parameters(model):
    """
    Count trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = AutoEncoderV2(num_filters=48, task='geometry', snr=10)
    print(f"AutoEncoderV2 Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 32, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
