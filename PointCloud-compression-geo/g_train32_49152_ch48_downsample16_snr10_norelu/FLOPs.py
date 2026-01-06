import os
import argparse
import subprocess

import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.utils.data as data
import TAE
from train import pad_collate_fn, pc_to_torch
from TAE import AWGNChannel
import pc_io
import thop

from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(
        prog='compress.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument(
        '--input_dir',
        help='Input directory.',type=str,default='./data/test/forhhy/rhetorician_block_32')
parser.add_argument(
    '--input_pattern',
    help='Mesh detection pattern.',type=str,default='*.ply')
parser.add_argument(
    '--output_dir',
    help='Output directory.',type=str,default='./PointCloud-compression-geo/output/forhhy/rhetorician_block32_torch_resnettranspose49152_norelu_ch48_down16_snr10_g2500_train32_nocolor_alpha090_model400_compressed')
parser.add_argument(
    '--checkpoint_dir',
    help='Directory where to save/load model checkpoints.',type=str,default='./model/block_32_norgb_ch48_downsample16_snr10_g2500_c0_resnettranspose49152_norelu_torchtest_alpha090')
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='Batch size.')
parser.add_argument(
    '--read_batch_size', type=int, default=1,
    help='Batch size for parallel reading.')
parser.add_argument(
    '--resolution',
    type=int, help='Dataset resolution.', default=32)
parser.add_argument(
    '--task', type=str, default='geometry',
    help='Compression tasks (geometry/color/geometry+color).')
parser.add_argument(
    '--num_filters', type=int, default=48,
    help='Number of filters per layer.')
parser.add_argument(
    '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
parser.add_argument(
        '--color_space', type=str, default='rgb',
        help='Color space type.')
parser.add_argument(
        '--network_type', type=str, default='unified',
        help='Neural network type.')
parser.add_argument(
        '--channels_last', action='store_true',
        help='Use channels last instead of channels first.')


args = parser.parse_args()

MODEL_FILE = os.path.join(args.checkpoint_dir, 'model_epoch_400.pth')# 300
# ae = torch.load(MODEL_FILE).cuda().eval()
ae = TAE.AutoEncoder(num_filters = args.num_filters, task = args.task, snr = 10)  # 根据你的模型参数进行调整

# 加载模型状态字典
state_dict = torch.load(MODEL_FILE)
ae.load_state_dict(state_dict)

# 移动模型到 GPU 并设置为评估模式
ae = ae.cuda()
ae.eval()



# 创建虚拟输入数据
dummy_input = torch.randn(1, 1, 32, 32, 32).cuda()  # 假设输入大小为 [1, 4, 32, 32, 32]
dummy_output = torch.randn(1, 48, 2, 2, 2).cuda()  # 假设输入大小为 [1, 4, 32, 32, 32]


# 计算 FLOPs 和参数量
flops_encoder, params_encoder = profile(ae.encoder, inputs=(dummy_input,),verbose=True)
flops_decoder, params_decoder = profile(ae.decoder, inputs=(dummy_output,),verbose=True)
flops_encoder, params_encoder = clever_format([flops_encoder, params_encoder], "%.3f")
flops_decoder, params_decoder = clever_format([flops_decoder, params_decoder], "%.3f")

print(f"FLOPs_encoder: {flops_encoder}")
print(f"FLOPs_decoder: {flops_decoder}")
print(f"Parameters_encoder: {params_encoder}")
print(f"Parameters_decoder: {params_decoder}")