import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

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
    def __init__(self, snr, bits=8, alpha=1.0, beta=1.0, nonlinearity='rapp', p=3.0, sat=1.0):
        super(ADDAChannel, self).__init__()
        self.awgn = AWGNChannel(snr)
        self.bits = bits
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nonlinearity
        self.p = p
        self.sat = sat
        self.inl_gamma = 0.01  # Fixed internal parameter for DAC INL

    def _tanh_model(self, x):
        # Model: y = alpha * tanh(beta * x)
        return self.alpha * torch.tanh(self.beta * x)

    def _rapp_model(self, x):
        # Rapp Model: V_out = V_in / (1 + (|V_in| / V_sat)^(2p))^(1/2p)
        num = x
        den = (1 + (torch.abs(x) / self.sat).pow(2 * self.p)).pow(1 / (2 * self.p))
        return num / den

    def forward(self, x):
        # 1. Quantization (DAC resolution limit)
        x = QuantizationLayer.apply(x, self.bits)

        # 2. INL (Integral Non-Linearity for DAC)
        # Cubic distortion: y = x + gamma * x^3
        x = x + self.inl_gamma * torch.pow(x, 3)

        # 3. Non-linearity (PA Saturation)
        if self.nonlinearity == 'rapp':
            x = self._rapp_model(x)
        else:
            x = self._tanh_model(x)
        
        # 4. Physical Channel (AWGN)
        x = self.awgn(x)
        
        return x



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
    def __init__(self, num_filters, task, snr, enable_adda=False, adda_bits=8, adda_alpha=1.0, adda_beta=1.0, nonlinearity='rapp', adda_p=3.0, adda_sat=1.0):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderNetwork(task, num_filters)
        if enable_adda:
            self.awgnchannel = ADDAChannel(
                snr, 
                bits=adda_bits, 
                alpha=adda_alpha, 
                beta=adda_beta,
                nonlinearity=nonlinearity,
                p=adda_p,
                sat=adda_sat
            )
        else:
            self.awgnchannel = AWGNChannel(snr)
        self.decoder = DecoderNetwork(task, num_filters)


    def forward(self, x):
        x = self.encoder(x)
        x = self.awgnchannel(x)
        x = self.decoder(x)
        return x