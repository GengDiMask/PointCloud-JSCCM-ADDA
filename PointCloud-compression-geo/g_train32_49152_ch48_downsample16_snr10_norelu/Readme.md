# 点云轻量化联合编码调制系统 (JCMPC)

本项目实现了基于论文 **"Lightweight Joint Coding-Modulation Optical Fiber Communication System for Point Cloud"** 的点云联合信源信道编码（JSCC）方案。

## 1. 项目简介 (Project Overview)

本项目提出了一种基于3D卷积神经网络的点云传输系统（JCMPC），旨在通过联合编码调制技术实现高效的点云光纤通信传输。
- **核心特性**: 采用双分支对称卷积神经网络结构，直接将点云体素块映射为模拟符号。
- **优势**: 相比传统分离式编码方案，本方案能有效避免“悬崖效应”，在信道条件恶化时展现出更平滑的性能下降，同时大幅降低解码端的计算复杂度（降低约80%）。
- **数据集**: 训练和验证使用大小为 $32 \times 32 \times 32$ 的体素化点云块，数据集总量为 49,152 个块。

## 2. 文件结构与功能 (File Structure)

主要代码文件功能如下：

| 文件名 | 功能描述 |
|--------|----------|
| `TAE.py` | **网络模型定义**。包含 Encoder（编码器）、Decoder（解码器）、AWGNChannel（加性高斯白噪声信道）以及用于处理样本不平衡的 Focal Loss 定义。网络采用了双分支对称卷积结构。 |
| `train_resume.py` | **网络训练脚本**。专门用于训练模型（支持从头训练或加载 checkpoint 继续训练）。默认配置：48个通道 (ch48)，SNR=10dB。 |
| `compress.py` | **编码（压缩）脚本**。加载训练好的模型，将输入的 .ply 点云块压缩为特征向量，并输出为 .txt 文件。 |
| `decompress.py` | **解压与实验验证脚本**。用于对压缩数据进行重建。支持常规 AWGN 信道仿真或 ADDA 补偿实验模式。 |
| `partition.py` | **分块工具**。将大场景或完整物体的点云分割为 $32 \times 32 \times 32$ 的标准小块，以便网络输入。 |
| `merge.py` | **合并工具**。将解压重建后的多个点云小块合并回一个完整的点云文件。 |
| `pc_io.py` | **IO 工具库**。包含点云文件的读取、写入、格式转换及辅助处理函数。 |
| `FLOPs.py` | **复杂度测试**。用于计算网络模型的计算量（FLOPs），评估轻量化程度。 |

## 3. 使用说明 (Usage Guide)

### 3.1 训练 (Training)
使用 `train_resume.py` 可以方便地恢复训练。
```bash
python train_resume.py --num_filters 48 --batch_size 64 --task geometry --checkpoint_dir ./model/block_32_...
```
- **参数说明**:
    - `--num_filters`: 卷积层滤波器数量，本项目中配置为 48。
    - `--lmbda_g`: 几何失真权重（Lambda），用于平衡码率和失真。
    - `--alpha`, `--gamma`: Focal Loss 的超参数，用于解决体素稀疏性问题。

### 3.2 压缩 (Compression)
使用 `compress.py` 对分块后的点云进行推理编码。
```bash
python compress.py --input_dir <输入ply文件夹> --output_dir <输出txt文件夹> --checkpoint_dir <模型路径>
```

### 3.3 解压与实验验证 (Decompression)
使用 `decompress.py` 对压缩数据进行重建。
- **输入**: 经过信道传输后的数据（或压缩脚本生成的 txt 文件）。
- **输出**: 重建后的 ply 点云文件。
- **注意**: 若进行 ADDA 实验，请开启 `--enable_adda` 参数。

```bash
python decompress.py --input_dir <压缩数据路径> --output_dir <重建输出路径> --enable_adda ...
```

### 3.4 辅助工具
1. **分块**: 先运行 `partition.py` 将原始点云切分为 $32 \times 32 \times 32$ 的块。
2. **合并**: 解压完成后，运行 `merge.py` 将块合并为完整点云以计算最终指标（D1-PSNR, D2-PSNR）。

## 4. 注意事项
- 本目录代码配置为 `g_train32_49152_ch48_downsample16_snr10_norelu`，即：
    - 训练块大小: 32
    - 数据量: 49152
    - 通道数: 48
    - 下采样倍率: 16 (基于网络结构推断)
    - 目标 SNR: 10dB
    - 激活函数: No ReLU (部分层或特定配置)
- 确保输入数据的维度与训练配置一致。

## 5. 链路损伤补偿研究方案 (训练方案设计)

针对开题报告中 **"基于一体化联合编码调制技术的链路损伤补偿研究"**，特别是为了弥补 ADDA（模数/数模转换器）的非线性效应和量化噪声，本部分提出一套针对性的训练方案。本方案仅针对点云**几何信息**进行训练。

### 5.1 总体思路

在 JSCCM 端到端训练框架中，将 ADDA 的物理损伤建模为可微的数学层（Differentiable Layer），嵌入到编码器（Tx）和解码器（Rx）之间。网络将在训练过程中自动学习：
1.  **编码器 (Tx)**: 学习 **预失真 (Pre-distortion)** 策略，主动抵消即将发生的非线性。
2.  **解码器 (Rx)**: 学习 **非线性均衡 (Post-equalization)**，从失真信号中恢复特征。

### 5.2 核心模型建模 (Channel Modeling)

需要在 `TAE.py` 中扩展 `AWGNChannel` 或新增 `ADDAChannel` 类。

#### A. ADDA 非线性模型 (Non-linearity)
ADDA 的传输特性曲线通常呈现 S 型饱和或非线性。为了更真实地模拟物理器件（如固态功率放大器 SSPA），本项目支持两种模型：
1.  **Rapp 模型 (Rapp Model)**：通信领域标准的功放行为模型，模拟平滑饱和特性。
    *   公式：$V_{out} = \frac{V_{in}}{\left(1 + \left(\frac{|V_{in}|}{V_{sat}}\right)^{2p}\right)^{\frac{1}{2p}}}$
2.  **Tanh 模型 (Tanh Model)**：简单的双曲正切饱和模型，作为 Baseline。
    *   公式：$V_{out} = \alpha \cdot \tanh(\beta \cdot V_{in})$

在 `TAE.py` 中已实现这两种模型，可通过 `--nonlinearity` 参数进行切换。

#### B. 量化噪声 (Quantization)
DAC/ADC 的有限分辨率（如 8-bit, 10-bit）引入量化误差。
-   **前向传播**: $y = \text{round}(x \cdot 2^{B-1}) / 2^{B-1}$
-   **反向传播**: 由于 `round` 函数不可导，需使用 **直通估计器 (Straight-Through Estimator, STE)**，即认为 $\frac{\partial y}{\partial x} = 1$。

### 5.3 详细执行步骤 (Execution Steps)

#### 步骤一：基准模型训练 (Baseline Training)

你可以选择从零开始训练，或者指定一个已有的 checkpoint 继续训练。

**A. 从零开始训练 (Start from scratch)**:
确保 `--resume_from` 为空或不指定（如果默认值为 None）。
```bash
# 默认从零开始 (需确保代码中 default=None)
python train_resume.py --checkpoint_dir ./model/baseline_snr10 --log_dir ./log/baseline

# 或者显式指定为空字符串
python train_resume.py --resume_from "" --checkpoint_dir ./model/baseline_snr10
```

**B. 从检查点恢复训练 (Resume from checkpoint)**:
```bash
# 指定要加载的 .pth 文件路径
python train_resume.py --resume_from ./model/baseline_snr10/model_epoch_400.pth --checkpoint_dir ./model/baseline_snr10_resumed
```

*注意：训练过程中会根据文件名（如 `model_epoch_400.pth`）自动提取起始 Epoch 编号，从而实现无缝衔接。*

#### 步骤二：ADDA 适应性训练 (Adaptation Training)
加载步骤一的权重，开启 ADDA 补偿模式进行微调。

本项目支持两种非线性模型：**Rapp 模型** (推荐，更符合物理特性) 和 **Tanh 模型** (Baseline)。

**A. 使用 Rapp 模型 (推荐)**
Rapp 模型是固态功率放大器 (SSPA) 的标准行为模型。
公式：$$ V_{out} = \frac{V_{in}}{\left(1 + \left(\frac{|V_{in}|}{V_{sat}}\right)^{2p}\right)^{\frac{1}{2p}}} $$
*   `--adda_p`: 平滑因子，控制线性区到饱和区的过渡陡峭程度（典型值 2~3）。
*   `--adda_sat`: 饱和电压 $V_{sat}$，限制最大输出幅度。

```bash
# 使用 Rapp 模型训练 (p=3.0, sat=1.0)
python train_resume.py --checkpoint_dir ./model/adda_rapp_snr10 \
    --resume_from ./model/baseline_snr10/model_epoch_400.pth \
    --enable_adda \
    --nonlinearity rapp \
    --adda_p 3.0 \
    --adda_sat 1.0 \
    --adda_bits 8 \
    --log_dir ./log/adda_rapp
```

**B. 使用 Tanh 模型 (Baseline)**
简单的双曲正切饱和模型。
公式：$$ V_{out} = \alpha \cdot \tanh(\beta \cdot V_{in}) $$
*   `--adda_alpha`: 幅度缩放因子。
*   `--adda_beta`: 输入增益因子。

```bash
# 使用 Tanh 模型训练
python train_resume.py --checkpoint_dir ./model/adda_tanh_snr10 \
    --resume_from ./model/baseline_snr10/model_epoch_400.pth \
    --enable_adda \
    --nonlinearity tanh \
    --adda_alpha 1.0 \
    --adda_beta 1.0 \
    --adda_bits 8 \
    --log_dir ./log/adda_tanh
```

### 5.4. 非线性模型对比 (Nonlinearity Config)

| 特性 | Rapp 模型 (`--nonlinearity rapp`) | Tanh 模型 (`--nonlinearity tanh`) |
| :--- | :--- | :--- |
| **物理意义** | 专为固态功放 (SSPA) 设计，精确模拟 AM/AM 转换 | 通用饱和函数，非特定物理模型 |
| **关键参数** | `p` (平滑度), `sat` (饱和电平) | `alpha` (幅度), `beta` (增益) |
| **行为** | 在低幅值区高度线性，接近饱和时平滑过渡 | 全局非线性，即使在低幅值区也有轻微失真 |
| **适用场景** | **高精度物理仿真**，模拟真实功放特性 | **理论研究**，作为简单的基准对比 |

> [!NOTE]
> **训练与测试的一致性**：
> 为了解决训练中不可导或梯度消失的问题，我们在代码内部实现了策略优化：
> *   **训练时 (Training)**: 使用 PyTorch 实现的可导 Rapp/Tanh 公式，支持自动微分。
> *   **测试时 (Testing/Decompress)**: 使用 Numpy 实现的**完全一致**的数学公式，确保测试结果真实反映物理特性。

#### 步骤三：模型测试与压缩 (Testing/Compression)
使用训练好的 ADDA 补偿模型进行点云压缩。
```bash
python compress.py \
    --enable_adda \
    --nonlinearity rapp \
    --adda_p 3.0 \
    --adda_sat 1.0 \
    --adda_bits 8 \
    --checkpoint_dir ./model/adda_rapp_snr10 \
    --input_dir ./data/test/blocks \
    --output_dir ./output/compressed_rapp
```

#### 步骤四：模型解压与验证 (Decompression)
解压数据并计算性能指标。注意解压时也需要开启 `enable_adda` 以应用正确的信道模型（如果是在仿真模式下）。
```bash
python decompress.py \
    --enable_adda \
    --adda_bits 8 \
    --adda_alpha 1.0 \
    --adda_beta 1.0 \
    --checkpoint_dir ./model/adda_adaptation \
    --input_dir ./output/compressed_adda \
    --output_dir ./output/decompressed_adda
```