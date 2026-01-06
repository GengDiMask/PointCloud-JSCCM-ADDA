import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import subprocess
import time
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.utils.data as data
import TAE
from train import pad_collate_fn, pc_to_torch
from TAE import AWGNChannel
import pc_io

parser = argparse.ArgumentParser(
        prog='compress.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument(
        '--input_dir',
        help='Input directory.',type=str,default='./data/test/forhhy/NP_supplemented/guanyin_block_32')# './data/test/forhhy/rhetorician_block_32'
parser.add_argument(
    '--input_pattern',
    help='Mesh detection pattern.',type=str,default='*.ply')
parser.add_argument(
    '--output_dir',
    help='Output directory.',type=str,default='./PointCloud-compression-geo/output/guanyin_block32_testtime_resnettranspose49152_norelu_ch48_down16_snr10_g2500_train32_nocolor_alpha090_model400_compressed')
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
# 原始:
parser.add_argument('--model_save_folder', help='Directory where to save/load trained models.')
parser.add_argument('--glob_input_path', help='Point clouds glob pattern to be compressed.')
parser.add_argument('--compressed_path', help='Compressed file saving directory.')
parser.add_argument('--load_scale', type=int, help='Input point cloud coordinate scale. [0, load_scale]', default=1)
parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
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
#

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
start_time = time.time()

DATA_FORMAT = 'channels_first' if not args.channels_last else 'channels_last'
args.input_dir = os.path.normpath(args.input_dir)# 'data/test/reddress_block_32'
len_input_dir = len(args.input_dir)# 27
assert os.path.exists(args.input_dir), "Input directory not found"
input_glob = os.path.join(args.input_dir, args.input_pattern)# 'data/test/reddress_block_32/*.ply'
files = pc_io.get_files(input_glob)
assert len(files) > 0, "No input files found"
filenames = [x[len_input_dir + 1:] for x in files]
output_files = [os.path.join(args.output_dir, x + '.txt') for x in filenames]
p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.task, args.resolution, args.channels_last)
points = pc_io.load_points(files, p_min, p_max, args.task, batch_size=args.read_batch_size)

vol_points = pad_collate_fn(points,task=args.task)
vol_points = vol_points.cuda()






# # CREATE COMPRESSED PATH
# if not os.path.exists(args.compressed_path):
#     os.makedirs(args.compressed_path)

# # READ INPUT FILES
# p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.load_scale)# p_min:array([0,0,0]) p_max:array([0,0,0]) dense_tensor_shap:array([1,1,1,1])
# files = pc_io.get_files(args.glob_input_path)
# filenames = np.array([os.path.split(x)[1] for x in files])

MODEL_FILE = os.path.join(args.checkpoint_dir, 'model_epoch_400.pth')# 300
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


# DO THE COMPRESS
with torch.no_grad():
    # for i in tqdm(range(len(filenames))):
    for i in range(len(filenames)):
        y=ae.encoder(vol_points[i])
        representation = torch.flatten(y)
        representation = representation.cpu().numpy()
        np.savetxt(output_files[i],representation,fmt = '%f', delimiter = ',')

end_time = time.time()
wall_time = end_time - start_time
print(f"Total execution time (Wall Time): {wall_time:.2f} seconds")
