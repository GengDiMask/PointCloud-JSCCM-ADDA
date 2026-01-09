#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decompression script
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

import torch
import argparse
import gzip
import logging
import multiprocessing
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import TAE
from tqdm import tqdm

import pc_io
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)

# 需不需要用那个类的还要另说
def AWGN_channel(x, snr):  # AWGN channel consider Peak power limitated system this SNR equal peak signal noise ratio
    [ batch_size,num_file,len_1, len_2,len_3] = x.shape
    x_power = np.sum(x ** 2) / (batch_size * num_file * len_1* len_2*len_3)
    n_power = x_power / (10 ** (snr / 10.0))
    n_power = np.sqrt(n_power)
    noise = np.random.randn(batch_size, num_file, len_1, len_2, len_3) *n_power ## change to guassian 
    return x + noise

def load_compressed_file(c_file):
    string = np.loadtxt(c_file)
    string = np.reshape(string,(args.num_filters,channel_num,channel_num,channel_num))
    return string


def load_compressed_files(files):
    files_len = len(files)
    # with multiprocessing.Pool() as p:
    #     logger.info('Loading data into memory (parallel reading)')
    #     list_file = p.imap(load_compressed_file, files)
    #     data = np.array(
    #         list(tqdm(list_file, total=files_len)))
    # logger.info('Loading data into memory (parallel reading)')
    files_list = files.tolist()
    data = []
    logger.info('Loading data into memory (parallel reading)')
    for file_name in files_list:
        data.append(load_compressed_file(file_name))
    data = np.reshape(data,(files_len,args.num_filters,channel_num,channel_num,channel_num))
    # data = np.stack(data,axis=0)
    # data = tf.convert_to_tensor(data)
    return data

def quantize_tensor(x):
    x = torch.round(x)  # 四舍五入
    x = x.to(dtype=torch.uint8)  # 转换为 uint8 类型
    return x

def ADDA_channel(x, snr, bits, alpha, beta):
    """
    Apply ADDA channel impairments using Numpy (Non-linearity -> Quantization -> AWGN).
    """
    # 1. Non-linearity (DAC saturation)
    # Model: y = alpha * tanh(beta * x)
    x = alpha * np.tanh(beta * x)
    
    # 2. Quantization (DAC resolution limit)
    scale = 2 ** (bits - 1)
    x = np.round(x * scale) / scale
    
    # 3. Physical Channel (AWGN)
    # Reuse the existing AWGN_channel function
    x = AWGN_channel(x, snr)
    
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='decompress.py',
        description='Decompress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--geo_dir',
        help='Geometry input directory.',type=str,default='data/test/blocks_32_rgb_longdress')
    parser.add_argument(
        '--geo_pattern',
        help='Geometry mesh detection pattern.',type=str,default='*.ply')
    parser.add_argument(
        '--input_dir',
        help='Input directory.',type=str,default='./PointCloud-compression-geo/output/guanyin_block32_testtime_resnettranspose49152_norelu_ch48_down16_snr10_g2500_train32_nocolor_alpha090_model400_compressed')
    parser.add_argument(
        '--input_pattern',
        help='Mesh detection pattern.',type=str,default='*.ply.txt')
    parser.add_argument(
        '--output_dir',
        help='Output directory.',type=str,default='./PointCloud-compression-geo/decompressed/guanyin_block32_torch_resnettranspose49152_norelu_ch48_down16_snr10_g2500_train32_nocolor_alpha090_model400_compressed_decompressed_testtime/')
    parser.add_argument(
        '--checkpoint_dir',
        help='Directory where to save/load model checkpoints.',type=str,default='./model/block_32_norgb_ch48_downsample16_snr10_g2500_c0_resnettranspose49152_norelu_torchtest_alpha090')
    parser.add_argument(
        '--model_name', type=str, default='model_epoch_400.pth',
        help='The filename of the model checkpoint.')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size.')
    # 没用到
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
        '--output_extension', default='.ply',
        help='Output extension.')
    parser.add_argument(
        '--color_space', type=str, default='rgb',
        help='Color space type.')
    parser.add_argument(
        '--network_type', type=str, default='base',
        help='Neural network type.')
    parser.add_argument(
        '--channels_last', action='store_true',
        help='Use channels last instead of channels first.')

    # ADDA Compensation Arguments
    parser.add_argument(
        '--enable_adda', action='store_true',
        help='Enable ADDA channel compensation training.')
    parser.add_argument(
        '--adda_bits', type=int, default=8,
        help='Quantization bits for ADDA.')
    parser.add_argument(
        '--adda_alpha', type=float, default=1.0,
        help='ADDA non-linearity parameter alpha.')
    parser.add_argument(
        '--adda_beta', type=float, default=1.0,
        help='ADDA non-linearity parameter beta.')

    args = parser.parse_args()

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first' if not args.channels_last else 'channels_last'

    args.input_dir = os.path.normpath(args.input_dir)
    len_input_dir = len(args.input_dir)
    assert os.path.exists(args.input_dir), "Input directory not found"

    input_glob = os.path.join(args.input_dir, args.input_pattern)
    files = pc_io.get_files(input_glob)
    assert len(files) > 0, "No input files found"

    if args.task == 'color':
        args.geo_dir = os.path.normpath(args.geo_dir)
        len_geo_dir = len(args.geo_dir)
        assert os.path.exists(args.geo_dir), "Geometry input directory not found"
        geo_glob = os.path.join(args.geo_dir, args.geo_pattern)
        geo_files = pc_io.get_files(geo_glob)
        assert len(geo_files) > 0, "No geometry input files found"
        p_min, p_max, _ = pc_io.get_shape_data(args.task, args.resolution, args.channels_last)
        points = pc_io.load_points(geo_files, p_min, p_max, args.task)

    filenames = [x[len_input_dir + 1:] for x in files]
    
    channel_num=2 
    compressed_data = load_compressed_files(files)
 
    x_shape = np.array([channel_num,channel_num,channel_num],np.uint16)
    y_shape = np.array([channel_num,channel_num,channel_num],np.uint16)

    MODEL_FILE = os.path.join(args.checkpoint_dir, args.model_name)
    # ae = torch.load(MODEL_FILE).cuda().eval()
    # ae = torch.load(MODEL_FILE).cuda().eval()
    ae = TAE.AutoEncoder(
        num_filters=args.num_filters,
        task=args.task, 
        snr=10, 
        enable_adda=args.enable_adda,
        adda_bits=args.adda_bits,
        adda_alpha=args.adda_alpha,
        adda_beta=args.adda_beta
    )  # 根据你的模型参数进行调整

    # 加载模型状态字典
    state_dict = torch.load(MODEL_FILE)
    ae.load_state_dict(state_dict)

    # 移动模型到 GPU 并设置为评估模式
    ae = ae.cuda()
    ae.eval()

    for snr in range(0,2,1):
        #args.output_dir = os.path.normpath(args.output_dir+'{}'.format(snr)+'dB')
        # a = os.path.normpath(args.output_dir + '{}'.format(snr) + 'dB')
        # output_files = [os.path.join(a, x + 'dB.ply') for x in filenames]
        if args.enable_adda:
            # Use Numpy implementations for ADDA channel
            code_input = ADDA_channel(
                compressed_data, 
                snr, 
                args.adda_bits, 
                args.adda_alpha, 
                args.adda_beta
            ).astype(np.float32)
        else:
            # Use original numpy AWGN channel
            code_input = AWGN_channel(compressed_data, snr).astype(np.float32)

        # codeWithNoise = compressed_data.astype(np.float32)


        len_files = len(files)
        i = 0
        if args.task == 'geometry':
            with torch.no_grad():
                for i in tqdm(range(len(filenames))):
                    output_dir1=os.path.normpath(args.output_dir+'{}'.format(snr)+'dB')
                    os.makedirs(output_dir1, exist_ok=True)
                    output_file=os.path.join(output_dir1, filenames[i] + 'dB.ply')
                    y=code_input[i]
                    y_tensor = torch.from_numpy(y).float()# 32
                    y_tensor = y_tensor.cuda()
                    
                    # Add batch dimension: (C,D,H,W) -> (1,C,D,H,W)
                    # Necessary because the model expects a batch dimension (N, ...)
                    y_tensor = y_tensor.unsqueeze(0) 
                    
                    y1 = ae.decoder(y_tensor)
                    
                    # Remove batch dimension for post-processing
                    y1 = y1.squeeze(0)

                    y1_quant=quantize_tensor(y1)
                    y1_quant_np = y1_quant.cpu().numpy()
                    # y1_np_quant=quantize_tensor(y1_np)
                    pa = np.argwhere(y1_quant_np[0]).astype('float32')# 找出非零元素的索引
                    pc_io.write_df(output_file, pc_io.pa_to_df(pa, args.task))
                    # b=0
        #     for ret, ori_file, output_file in zip(result, files, output_files):
        #         logger.info(f'{i + 1}/{len_files} - Writing {ori_file} to {output_file}')
        #         output_dir, _ = os.path.split(output_file)
        #         os.makedirs(output_dir, exist_ok=True)
        #         # Remove the geometry channel
        #         pa = np.argwhere(ret['x_hat_quant'][0]).astype('float32')# 找出非零元素的索引
        #         pc_io.write_df(output_file, pc_io.pa_to_df(pa, args.task))
        #         i += 1
        #         # get_flops_params()#计算FLOPs
        # elif args.task == 'color':

        # elif args.task == 'geometry+color':


        
        
        # # DO THE COMPRESS
        # with torch.no_grad():
        #     for i in tqdm(range(len(filenames))):
        #         y=ae.encoder(vol_points[i])
        #         representation = torch.flatten(y)
        #         representation = representation.cpu().numpy()
        #         np.savetxt(output_files[i],representation,fmt = '%f', delimiter = ',')
