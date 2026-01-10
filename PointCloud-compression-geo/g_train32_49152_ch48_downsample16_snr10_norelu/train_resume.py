import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
# import torch.utils.data as Data
import re
import torch.utils.data as data
import TAE
import pc_io
import warnings
import logging
import random
import os
import torch.optim.lr_scheduler as lr_scheduler
print("Current Working Directory:", os.getcwd())

warnings.filterwarnings('ignore', category=FutureWarning)

# Set random seeds for reproducibility
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
RANDOM_SEED = 42
set_seeds(RANDOM_SEED)


'''pad_collate_fn and pc_to_torch 对几何应该是没问题,对几何加颜色还需要修改'''
def pad_collate_fn(batch,task):
    dense_tensor_shape = torch.Size([1, 32, 32, 32])  # 目标张量的形状
    padded_batch = []

    for points in batch:
        # 使用 pc_to_torch 函数填充点云数据
        sparse_tensor = pc_to_torch(points, dense_tensor_shape, task=task)
        dense_tensor = sparse_tensor.to_dense()  # 将稀疏张量转换为密集张量
        padded_batch.append(dense_tensor)

    # 将每个 batch 中的点云块拼接成 tensor
    return torch.stack(padded_batch)

def pc_to_torch(points, dense_tensor_shape, task, channels_last=False):
    # 将 points 转换为 torch.Tensor
    x = torch.tensor(points)

    # 填充操作：channels_last=False 时在第一个维度填充，channels_last=True 时在最后一维填充
    if not channels_last:
        # 在第一列前面手动添加一列0
        geo_indices = torch.cat([torch.zeros((x.shape[0], 1)), x[:, :3]], dim=1)
    else:
        # 在最后一列后面填充一列0
        geo_indices = torch.cat([x[:, :3], torch.zeros((x.shape[0], 1))], dim=1)

    if task == 'geometry':
        indices = geo_indices.long()  # 转为整型索引
        values = torch.ones_like(x[:, 0])  # 创建与点云数量相同的值
    else:
        # 填充颜色信息并生成索引
        r_indices = torch.cat([torch.ones((x.shape[0], 1)), x[:, :3]], dim=1)  # 前面填充1
        g_indices = torch.cat([torch.ones((x.shape[0], 1)) * 2, x[:, :3]], dim=1)  # 前面填充2
        b_indices = torch.cat([torch.ones((x.shape[0], 1)) * 3, x[:, :3]], dim=1)  # 前面填充3

        # 拼接索引
        indices = torch.cat([geo_indices, r_indices, g_indices, b_indices], dim=0).long()
        
        # 拼接对应的值
        values = torch.cat([torch.ones_like(x[:, 0]), x[:, 3], x[:, 4], x[:, 5]], dim=0)

    # 创建稀疏张量
    st = torch.sparse_coo_tensor(indices.t(), values, size=dense_tensor_shape)

    return st
def train():
    """Train the model."""
    resume_from = args.resume_from
    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(
        args.task, args.resolution, args.channels_last)
    files = pc_io.get_files(args.train_glob)
    perm = np.random.permutation(len(files))
    points = pc_io.load_points(files[perm][:args.num_data], p_min, p_max, args.task)

    points_train = points[:-args.num_val]
    points_val = points[-args.num_val:]

    enable_adda = not args.disable_adda
    ae = TAE.AutoEncoder(
        num_filters=args.num_filters, 
        task=args.task, 
        snr=10,
        enable_adda=enable_adda,
        adda_bits=args.adda_bits,
        adda_alpha=args.adda_alpha,
        adda_beta=args.adda_beta,
        nonlinearity=args.nonlinearity,
        p=args.adda_p,
        sat=args.adda_sat,
        dnl_sigma=args.dnl_sigma,
        inl_gamma=args.inl_gamma
    ).cuda()
    criterion = TAE.get_loss().cuda()

    optimizer = torch.optim.Adam(ae.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = data.DataLoader(
        points_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,  # Windows doesn't support multiprocessing with lambda
        collate_fn=lambda batch: pad_collate_fn(batch, task=args.task)
    )
    val_loader = data.DataLoader(
        points_val, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1,  # Windows doesn't support multiprocessing with lambda
        collate_fn=lambda batch: pad_collate_fn(batch, task=args.task)
    )

    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from)
        ae.load_state_dict(checkpoint)
        # Extracting epoch number from file name
        match = re.search(r"model_epoch_(\d+)\.pth", resume_from)
        if match:
            start_epoch = int(match.group(1))
        logger.info(f"Model loaded from {resume_from}, resuming from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, args.max_steps):
        ae.train()
        total_loss = 0
        count = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            outputs = ae(batch_data)

            geo_x = batch_data[:, 0] if not args.channels_last else batch_data[:, :, :, :, 0]
            if args.task == 'geometry':
                loss = criterion(outputs, batch_data) / geo_x.numel()
                loss = args.lmbda_g * loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{args.max_steps}, Train Loss: {avg_train_loss}")

        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(ae, val_loader, device, criterion)
            
            # Save to log file in dedicated logging folder
            log_dir = args.log_dir
            log_path = os.path.join(log_dir, 'train_log.txt')
            os.makedirs(log_dir, exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}\n")
            logger.info(f"Log updated at {log_path}")

        if (epoch + 1) % args.save_checkpoints_steps == 0:
            save_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(ae.state_dict(), save_path)
            logger.info(f"Model checkpoint saved at {save_path}")

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)

            geo_x = batch_data[:, 0] if not args.channels_last else batch_data[:, :, :, :, 0]
            if args.task == 'geometry':
                loss = criterion(outputs, batch_data) / geo_x.numel()
                loss = args.lmbda_g * loss

            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss
################################################################################
# Script
################################################################################
if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--train_glob',
        help='Glob pattern for identifying training data.',type=str,default='../code/data/train/blocks_32_rgb_49152/*.ply')

    parser.add_argument(
        '--checkpoint_dir',
        help='Directory where to save model checkpoints.',type=str,default='./model/baseline_snr10_nocur')

    parser.add_argument(
        '--log_dir',
        help='Directory where to save training logs.',type=str,default='./log/zero')

    parser.add_argument(
        '--resume_from',
        help='Path to a model checkpoint to resume training from.',type=str,default='./model/baseline_snr10/model_epoch_400.pth')

    parser.add_argument(#blocksize
        '--resolution',
        type=int, help='Dataset resolution.', default=32)

    parser.add_argument(
        '--num_data', type=int, default=None,
        help='Number of total data we want to use (-1: use all data).')

    parser.add_argument(
        '--num_val', type=int, default=4864,#64
        help='Number of validation data we want to use')

    # 没用到
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Report bitrate and distortion when training.')

    # 没用到
    parser.add_argument(
        '--no_additional_metrics', action='store_true',
        help='Report additional metrics when training.')
    
    # model每10个epoch保存一次
    parser.add_argument(
        '--save_checkpoints_steps', type=int, default=10,
        help='Save checkpoints every n steps during training.')

    # 没用到
    parser.add_argument(
        '--keep_checkpoint_max', type=int, default=1,
        help='Maximum number of checkpoint files to keep.')

    # 每多少个epoch进行一次验证 
    parser.add_argument(
        '--val_interval', type=int, default=1,
        help='Epoch interval between validation and logging.')

    # 没用到
    parser.add_argument(
        '--save_summary_steps', type=int, default=100,
        help='Save summaries every n steps.')

    parser.add_argument(
        '--task', type=str, default='geometry',
        help='Compression tasks (geometry/color/geometry+color).')

    parser.add_argument(
        '--num_filters', type=int, default=48,#32 控制压缩率
        help='Number of filters per layer.')

    parser.add_argument(
        '--batch_size', type=int, default=64,#16
        help='Batch size for training.')


    parser.add_argument(
        '--lmbda_g', type=float, default=2500,
        help='Lambda (geometry) for rate-distortion tradeoff.')

    parser.add_argument(
        '--lmbda_c', type=float, default=0,
        help='Lambda (color) for rate-distortion tradeoff.')

    parser.add_argument(
        '--alpha', type=float, default=0.9,
        help='Focal loss alpha.')

    parser.add_argument(
        '--gamma', type=float, default=2.0,
        help='Focal loss gamma.')

    # 训练的epoch数
    parser.add_argument(
        '--max_steps', type=int, default=1000,
        help='Train up to this number of steps.')

    parser.add_argument(
        '--loss_type', type=str, default='l2',
        help='Color loss type.')

    parser.add_argument(
        '--ssim_filter', type=int, default=6,
        help='Filter size for ssim loss.')

    parser.add_argument(
        '--loss_weight', type=str, default='f,1,1,1',
        help='Loss weights for different channels.')

    parser.add_argument(
        '--channels_last', action='store_true',
        help='Use channels last instead of channels first.')
        
    # ADDA Compensation Arguments
    # ADDA Compensation Arguments (Non-linearity, INL/DNL)
    parser.add_argument(
        '--disable_adda', action='store_true',
        help='Disable ADDA channel compensation training (ADDA is ENABLED by default).')
    
    parser.add_argument(
        '--nonlinearity',
        help="Type of nonlinearity: 'rapp' (PA), 'tanh' (Baseline), or 'none' (Pure ADC Mode). Default is 'none'.",
        type=str, default='none', choices=['rapp', 'tanh', 'none'])
        
    parser.add_argument(
        '--dnl_sigma',
        help="Std dev of DNL noise (Quantization threshold jitter). Default 0.01.",
        type=float, default=0.01)
        
    parser.add_argument(
        '--inl_gamma',
        help="Coefficient for DAC Integral Non-Linearity (INL). Default 0.01.",
        type=float, default=0.01)

    parser.add_argument(
        '--adda_bits', type=int, default=8,
        help='Quantization bits for ADDA.')

    parser.add_argument(
        '--adda_alpha', type=float, default=1.0,
        help='ADDA non-linearity parameter alpha.')

    parser.add_argument(
        '--adda_beta', type=float, default=1.0,
        help='ADDA non-linearity parameter beta.')
        

    parser.add_argument(
        '--adda_p', type=float, default=3.0,
        help='Smoothness parameter p for Rapp model.')
    parser.add_argument(
        '--adda_sat', type=float, default=1.0,
        help='Saturation voltage V_sat for Rapp model.')

    args = parser.parse_args()

    # os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    assert args.resolution > 0, 'resolution must be positive'
    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first' if not args.channels_last else 'channels_last'

    train() 