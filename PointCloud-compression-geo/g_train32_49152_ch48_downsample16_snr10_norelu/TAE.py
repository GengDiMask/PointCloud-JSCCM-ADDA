import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

# class AWGNChannel:
#     def __init__(self, snr):
#         self.snr = snr

#     def apply(self, x):
#         batch_size, length = x.shape
#         x_power = torch.sum(torch.mul(x, x)) / (batch_size * length)
#         n_power = x_power / (10 ** (self.snr / 10.0))
#         n_power = torch.sqrt(n_power)
#         noise = torch.randn(batch_size, length, device=x.device) * n_power
#         return x + noise
#     def __call__(self, x):
#         return self.apply(x)
# 不确定是否对,后面还需要再看
class AWGNChannel:
    def __init__(self, snr):
        self.snr = snr

    def apply(self, x):
        # 获取输入张量的形状和总元素数
        shape = x.shape
        num_elements = torch.numel(x)

        # 计算信号的总功率除以元素数量，得到每个符号的平均功率
        x_power = torch.sum(x ** 2) / num_elements

        # 计算噪声功率
        n_power = x_power / (10 ** (self.snr / 10.0))

        # 生成与x形状相同的高斯噪声
        noise = torch.randn(shape, device=x.device) * torch.sqrt(n_power)

        # 将噪声添加到信号上并返回
        return x + noise

    def __call__(self, x):
        return self.apply(x)


# Quantization Layer with Straight-Through Estimator (STE)
class QuantizationLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        scale = 2 ** (bits - 1)
        return torch.round(x * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient flows through as if identity
        return grad_output, None


class ADDAChannel(nn.Module):
    def __init__(self, snr, bits=8, alpha=1.0, beta=1.0):
        super(ADDAChannel, self).__init__()
        self.awgn = AWGNChannel(snr)
        self.bits = bits
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x):
        # 1. Non-linearity (DAC/ADC saturation simulation)
        # Model: y = alpha * tanh(beta * x)
        x = self.alpha * torch.tanh(self.beta * x) 
        
        # 2. Quantization (DAC resolution limit)
        x = QuantizationLayer.apply(x, self.bits)
        
        # 3. Physical Channel (AWGN)
        x = self.awgn(x)
        
        return x



# 需要自己写
# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target):
#         '''
#         Input:
#             pred: reconstructed point cloud (B, N, 3)
#             target: origin point cloud (B, CxN, 3)
#             bottleneck: 
#         '''
#         d, d_normals = chamfer_distance(pred, target)
#         loss = d

#         return loss

class get_loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.90):
        super(get_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # Clip prediction tensor to prevent log(0)
        pt_1 = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))

        pt_1 = torch.clamp(pt_1, 1e-3, 0.999)
        pt_0 = torch.clamp(pt_0, 1e-3, 0.999)

        # Calculate focal loss components
        loss_1 = -self.alpha * torch.pow(1. - pt_1, self.gamma) * torch.log(pt_1)
        loss_0 = -(1 - self.alpha) * torch.pow(pt_0, self.gamma) * torch.log(1. - pt_0)

        return torch.sum(loss_1) + torch.sum(loss_0)
# # simple cnn
# class EncoderNetwork(nn.Module):
#     def __init__(self, num_filters):
#         super(EncoderNetwork, self).__init__()
#         self.conv1 = nn.Conv3d(1, num_filters, kernel_size=9, stride=2, padding=4)
#         self.conv2 = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
#         self.conv3 = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
#         self.conv4 = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # [16, num_filters, 16, 16, 16]
#         x = F.relu(self.conv2(x))  # [16, num_filters, 8, 8, 8]
#         x = F.relu(self.conv3(x))  # [16, num_filters, 4, 4, 4]
#         x = F.relu(self.conv4(x))  # [16, num_filters, 2, 2, 2]
#         return x

# class DecoderNetwork(nn.Module):
#     def __init__(self, task, num_filters):
#         super(DecoderNetwork, self).__init__()
#         if task == 'geometry':
#             ch = 1
#         elif task == 'color':
#             ch = 3
#         elif task == 'geometry+color':
#             ch = 4

#         self.deconv1 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv2 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv3 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv4 = nn.ConvTranspose3d(num_filters, ch, kernel_size=9, stride=2, padding=4, output_padding=1)

#     def forward(self, x):
#         x = F.relu(self.deconv1(x))  # [16, num_filters, 4, 4, 4]
#         x = F.relu(self.deconv2(x))  # [16, num_filters, 8, 8, 8]
#         x = F.relu(self.deconv3(x))  # [16, num_filters, 16, 16, 16]
#         x = torch.sigmoid(self.deconv4(x))  # [16, ch, 32, 32, 32]
#         return x


# # resnet

# class ResidualBlock(nn.Module):
#     def __init__(self, num_filters):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)

#     def forward(self, x):
#         y1 = F.relu(self.conv1(x))
#         y1 = F.relu(self.conv2(y1))

#         y2 = F.relu(self.conv3(x))
#         y2 = F.relu(self.conv4(y2))
#         y2 = F.relu(self.conv5(y2))

#         output = y1 + y2
#         return output

# class EncoderNetwork(nn.Module):
#     def __init__(self, task, num_filters):
#         super(EncoderNetwork, self).__init__()
#         if task == 'geometry':
#             ch = 1
#         elif task == 'color':
#             ch = 3
#         elif task == 'geometry+color':
#             ch = 4
#         self.conv = nn.Conv3d(ch, num_filters, kernel_size=9, stride=2, padding=4)
#         self.conv_int = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
#         self.convout = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=4, padding=2, bias=False)
#         # He Uniform initialization for the last layer in the encoder
#         init.kaiming_uniform_(self.convout.weight, nonlinearity='relu')
#         self.residual_block = ResidualBlock(num_filters)

#     def forward(self, x):
#         x = F.relu(self.conv(x))
#         x = self.residual_block(x)
#         x = F.relu(self.conv_int(x))
#         x = self.residual_block(x)
#         x = F.relu(self.convout(x))
#         return x

# class DecoderNetwork(nn.Module):
#     def __init__(self, task, num_filters):
#         super(DecoderNetwork, self).__init__()
#         if task == 'geometry':
#             ch = 1
#         elif task == 'color':
#             ch = 3
#         elif task == 'geometry+color':
#             ch = 4

#         self.conv1 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=4, padding=1, output_padding=1)
#         self.conv2 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.conv3 = nn.ConvTranspose3d(num_filters, ch, kernel_size=9, stride=2, padding=4, output_padding=1)
#         # Glorot Uniform initialization for the last layer in the decoder
#         init.xavier_uniform_(self.conv3.weight)
#         self.residual_block = ResidualBlock(num_filters)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.residual_block(x)
#         x = F.relu(self.conv2(x))
#         x = self.residual_block(x)
#         x = torch.sigmoid(self.conv3(x))
#         return x



# class AutoEncoder(nn.Module):
#     def __init__(self, num_filters, task, snr):
#         super(AutoEncoder, self).__init__()
#         self.encoder = EncoderNetwork(task, num_filters)
#         self.awgnchannel = AWGNChannel(snr)
#         self.decoder = DecoderNetwork(task, num_filters)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.awgnchannel(x)
#         x = self.decoder(x)
#         return x


# # improved version
# class AnalysisBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=(2, 2, 2), dropout_rate=0.1):
#         super(AnalysisBlock, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn3 = nn.BatchNorm3d(out_channels)
#         self.activation = nn.ReLU(inplace=False)
#         self.dropout = nn.Dropout3d(dropout_rate)

#     def forward(self, x):
        
#         x = self.activation(self.conv1(x))
#         residual = x
#         x = self.activation(self.conv2(x))
#         x = self.activation(self.conv3(x))
#         x1 = x + residual
#         x1 = self.dropout(x1)
#         return x1

# class SynthesisBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, upsample_rate=2, dropout_rate=0.1):
#         super(SynthesisBlock, self).__init__()
#         if upsample_rate == 4:
#             stride = 4
#             output_padding = 3  # 根据需要可能调整，取决于padding和kernel size
#         else:
#             stride = 2
#             output_padding = 1

#         self.deconv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=1, output_padding=output_padding)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.deconv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.deconv3 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1)
#         self.bn3 = nn.BatchNorm3d(out_channels)
#         self.activation = nn.ReLU(inplace=False)
#         self.dropout = nn.Dropout3d(dropout_rate)

#     def forward(self, x):
        
#         x = self.activation(self.deconv1(x))
#         residual = x
#         x = self.activation(self.deconv2(x))
#         x = self.activation(self.deconv3(x))
#         x1 = x + residual
#         x1 = self.dropout(x1)
#         return x1


# class AnalysisTransformProgressiveV2(nn.Module):
#     def __init__(self, task, filters):
#         super(AnalysisTransformProgressiveV2, self).__init__()
#         if task == 'geometry':
#             ch = 1
#         elif task == 'color':
#             ch = 3
#         elif task == 'geometry+color':
#             ch = 4
#         self.block1 = AnalysisBlock(ch, filters // 4)
#         self.block2 = AnalysisBlock(filters // 4, filters // 2)
#         # Adjust stride to 4x downsampling for the final block
#         self.block3 = AnalysisBlock(filters // 2, filters, stride=(4, 4, 4))
#         self.final_conv = nn.Conv3d(filters, filters, (3, 3, 3), padding=1, bias=False)
#         # He Uniform initialization for the last layer in the encoder
#         init.kaiming_uniform_(self.final_conv.weight, nonlinearity='relu')
#         self.activation = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.activation(self.final_conv(x))
#         return x

# class SynthesisTransformProgressiveV2(nn.Module): 
#     def __init__(self, task, filters):
#         super(SynthesisTransformProgressiveV2, self).__init__()
#         if task == 'geometry':
#             ch = 1
#         elif task == 'color':
#             ch = 3
#         elif task == 'geometry+color':
#             ch = 4
#         self.block1 = SynthesisBlock(filters, filters, upsample_rate=4)
#         self.block2 = SynthesisBlock(filters, filters//2)
#         self.block3 = SynthesisBlock(filters//2, filters//4)
#         self.final_deconv = nn.ConvTranspose3d(filters//4, ch, (3, 3, 3), padding=1)
#         # Glorot Uniform initialization for the last layer in the decoder
#         init.xavier_uniform_(self.final_deconv.weight)
        

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = torch.sigmoid(self.final_deconv(x))
#         return x

# class AutoEncoder(nn.Module):
#     def __init__(self, num_filters, task, snr):
#         super(AutoEncoder, self).__init__()
#         self.encoder = AnalysisTransformProgressiveV2(task, num_filters)
#         self.awgnchannel = AWGNChannel(snr)
#         self.decoder = SynthesisTransformProgressiveV2(task, num_filters)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.awgnchannel(x)
#         x = self.decoder(x)
#         return x

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = F.relu(self.conv1(x))
        y1 = F.relu(self.conv2(y1))

        y2 = F.relu(self.conv3(x))
        y2 = F.relu(self.conv4(y2))
        y2 = F.relu(self.conv5(y2))

        output = y1 + y2
        return output

class EncoderNetwork(nn.Module):
    def __init__(self, task, num_filters):
        super(EncoderNetwork, self).__init__()
        if task == 'geometry':
            ch = 1
        elif task == 'color':
            ch = 3
        elif task == 'geometry+color':
            ch = 4
        self.conv = nn.Conv3d(ch, num_filters, kernel_size=9, stride=2, padding=4)
        self.conv_int = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        self.convout = nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=4, padding=2, bias=False)
        # He Uniform initialization for the last layer in the encoder without ReLU
        init.kaiming_uniform_(self.convout.weight, nonlinearity='linear')
        self.residual_block = ResidualBlock(num_filters)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.residual_block(x)
        x = F.relu(self.conv_int(x))
        x = self.residual_block(x)
        x = self.convout(x)
        return x
class TransposedResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(TransposedResidualBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y1 = F.relu(self.deconv1(x))
        y1 = F.relu(self.deconv2(y1))

        y2 = F.relu(self.deconv3(x))
        y2 = F.relu(self.deconv4(y2))
        y2 = F.relu(self.deconv5(y2))

        output = y1 + y2
        return output

class DecoderNetwork(nn.Module):
    def __init__(self, task, num_filters):
        super(DecoderNetwork, self).__init__()
        if task == 'geometry':
            ch = 1
        elif task == 'color':
            ch = 3
        elif task == 'geometry+color':
            ch = 4

        self.conv1 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose3d(num_filters, ch, kernel_size=9, stride=2, padding=4, output_padding=1)
        # Glorot Uniform initialization for the last layer in the decoder
        init.xavier_uniform_(self.conv3.weight)
        self.residual_block = TransposedResidualBlock(num_filters)  # Use transposed residual block here

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.residual_block(x)  # Using the modified transposed residual block
        x = F.relu(self.conv2(x))
        x = self.residual_block(x)
        x = torch.sigmoid(self.conv3(x))
        return x
class AutoEncoder(nn.Module):
    def __init__(self, num_filters, task, snr, enable_adda=False, adda_bits=8, adda_alpha=1.0, adda_beta=1.0):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderNetwork(task, num_filters)
        if enable_adda:
            self.awgnchannel = ADDAChannel(snr, bits=adda_bits, alpha=adda_alpha, beta=adda_beta)
        else:
            self.awgnchannel = AWGNChannel(snr)
        self.decoder = DecoderNetwork(task, num_filters)


    def forward(self, x):
        x = self.encoder(x)
        x = self.awgnchannel(x)
        x = self.decoder(x)
        return x