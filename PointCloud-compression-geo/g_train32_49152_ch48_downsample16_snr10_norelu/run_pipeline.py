#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化点云处理流程脚本 (Automated Point Cloud Processing Pipeline)

功能：自动执行 压缩(compress) -> 解压(decompress) -> 合并(merge) 流程
使用：修改下方 CONFIG 部分的参数，然后运行本脚本即可

注意：需从项目根目录 (E:\\PointCloud\\code\\PC-ADDA) 运行本脚本
      运行示例：python PointCloud-compression-geo/g_train32_49152_ch48_downsample16_snr10_norelu/run_pipeline.py
"""

import os
import subprocess
import time

# ================================================================================
# CONFIG - 在此修改参数
# ================================================================================

# 脚本所在目录 (相对于项目根目录)
# 由于脚本现在就在该目录下，如果从根目录运行，路径如下：
SCRIPT_DIR = "./PointCloud-compression-geo/g_train32_49152_ch48_downsample16_snr10_norelu"

# --- 通用参数 ---
CHECKPOINT_DIR = "./model/block_32_norgb_ch48_downsample16_snr10_g2500_c0_resnettranspose49152_norelu_torchtest_alpha090"
NUM_FILTERS = 48
TASK = "geometry"  # geometry / color / geometry+color
RESOLUTION = 32

# --- ADDA 相关参数 (如不需要可设为 False) ---
ENABLE_ADDA = False
ADDA_BITS = 8
ADDA_ALPHA = 1.0
ADDA_BETA = 1.0

# --- 输入/输出路径 ---
# 1. 压缩阶段：分块好的点云 -> 压缩后的 txt
INPUT_BLOCKS_DIR = "./data/test/forhhy/NP_supplemented/guanyin_block_32"
COMPRESSED_OUTPUT_DIR = "./PointCloud-compression-geo/output/guanyin_compressed"

# 2. 解压阶段：压缩后的 txt -> 解压后的点云块
DECOMPRESSED_OUTPUT_DIR = "./PointCloud-compression-geo/decompressed/guanyin_decompressed"

# 3. 合并阶段：解压后的点云块 -> 完整点云
ORIGINAL_PC_DIR = "./data/test/forhhy/NP_supplemented/guanyin"  # 用于获取原始文件名
MERGED_OUTPUT_DIR = "./PointCloud-compression-geo/merged/guanyin_merged"

# ================================================================================
# 以下代码无需修改
# ================================================================================

def run_command(cmd, step_name):
    """运行命令并打印状态"""
    print(f"\n{'='*60}")
    print(f"[{step_name}] 开始执行...")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=False)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[{step_name}] ✓ 完成! 耗时: {elapsed:.2f}秒")
    else:
        print(f"[{step_name}] ✗ 失败! 返回码: {result.returncode}")
        exit(1)

def main():
    print("\n" + "="*60)
    print("点云处理流程自动化脚本")
    print("="*60)
    
    # 1. 压缩 (Compress)
    compress_cmd = [
        "python", f"{SCRIPT_DIR}/compress.py",
        "--input_dir", INPUT_BLOCKS_DIR,
        "--output_dir", COMPRESSED_OUTPUT_DIR,
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--num_filters", str(NUM_FILTERS),
        "--task", TASK,
        "--resolution", str(RESOLUTION),
    ]
    if ENABLE_ADDA:
        compress_cmd.extend([
            "--enable_adda",
            "--adda_bits", str(ADDA_BITS),
            "--adda_alpha", str(ADDA_ALPHA),
            "--adda_beta", str(ADDA_BETA),
        ])
    run_command(compress_cmd, "Step 1: Compress")

    # 2. 解压 (Decompress)
    decompress_cmd = [
        "python", f"{SCRIPT_DIR}/decompress.py",
        "--input_dir", COMPRESSED_OUTPUT_DIR,
        "--output_dir", DECOMPRESSED_OUTPUT_DIR,
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--num_filters", str(NUM_FILTERS),
        "--task", TASK,
        "--resolution", str(RESOLUTION),
    ]
    if ENABLE_ADDA:
        decompress_cmd.extend([
            "--enable_adda",
            "--adda_bits", str(ADDA_BITS),
            "--adda_alpha", str(ADDA_ALPHA),
            "--adda_beta", str(ADDA_BETA),
        ])
    run_command(decompress_cmd, "Step 2: Decompress")

    # 3. 合并 (Merge)
    merge_cmd = [
        "python", f"{SCRIPT_DIR}/merge.py",
        "--ori_dir", ORIGINAL_PC_DIR,
        "--div_dir", DECOMPRESSED_OUTPUT_DIR,
        "--output_dir", MERGED_OUTPUT_DIR,
        "--resolution", str(RESOLUTION),
        "--task", TASK,
    ]
    run_command(merge_cmd, "Step 3: Merge")

    print("\n" + "="*60)
    print("🎉 所有步骤执行完成!")
    print(f"最终输出目录: {MERGED_OUTPUT_DIR}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
