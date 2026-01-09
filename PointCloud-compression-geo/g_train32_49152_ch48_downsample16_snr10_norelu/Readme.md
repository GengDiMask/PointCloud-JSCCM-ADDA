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
为了实现物理层面的精确优化，本项目对 **"Digital -> Analog -> PA -> Channel"** 的完整信号转换链路进行了精细建模。
新的信号处理顺序如下（物理逆序修正）：
**Quantization (DAC) -> INL (DAC Distortion) -> Nonlinearity (PA) -> AWGN (Channel)**

本项目包含以下两个核心非线性组件：

1.  **DAC 积分非线性 (Integral Non-Linearity, INL)** [**内置强制**]:
    *   为了针对真实 DAC 的非理性进行优化，代码中**内置并强制**了一个系数为 $\gamma=0.01$ 的三次谐波失真。
    *   公式：$$ V_{dac} = V_{quantized} + 0.01 \cdot V_{quantized}^3 $$
    *   该失真模拟了 DAC 转换过程中的本征非线性误差。
    *   **相关研究**: 多项式模型是描述 DAC/ADC INL 的经典方法 [1]；近年来，利用神经网络来建模和补偿 INL 已成为热点研究 [2][3]。

2.  **PA 功率放大器模型 (Power Amplifier Model)** [可选]:
    支持两种模型，作用于经过 DAC/INL 失真后的模拟信号：
    *   **Rapp 模型 (Rapp Model, 推荐)**：
        *   **来源**: C. Rapp, "Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal for a digital sound broadcasting system," in *Proc. ECSC*, 1991.
        *   **适用性**: 固态功率放大器 (SSPA) 的标准行为模型。
        *   公式：$$ V_{out} = \frac{V_{in}}{\left(1 + \left(\frac{|V_{in}|}{V_{sat}}\right)^{2p}\right)^{\frac{1}{2p}}} $$
        *   **参数说名**: 
            *   $V_{sat}$: **饱和电压** (Saturation Voltage)，即功放能输出的最大电压幅值。
            *   $p$: **平滑因子** (Smoothness Factor)，控制线性区到饱和区的过渡陡峭程度。
    *   **Tanh 模型 (Tanh Model, Baseline)**：
        *   **适用性**: 简单的双曲正切饱和模型。
        *   公式：$$ V_{out} = \alpha \cdot \tanh(\beta \cdot V_{in}) $$

在 `TAE.py` 中，信号会依次经过 `Quantization` -> `INL` -> `Rapp/Tanh` -> `AWGN`，确保神经网络能学习到对抗全链路非线性的能力。

#### 参考文献 (References)
[1] F. Maloberti, *Data Converters*, Springer, 2007. (Classic reference on ADC/DAC non-idealities including INL/DNL)

[2] J. Mullrich et al., "Data Converter Nonlinearity in Automotive Radar: Modeling, Estimation and Compensation," *IEEE Access*, 2022. (Discusses polynomial modeling of INL)

[3] K. Hornik et al., "Multilayer Feedforward Networks are Universal Approximators," *Neural Networks*, 1989. (Theoretical basis for NN compensation)

[4] Y. LeCun et al., "Deep Learning," *Nature*, 2015. (General deep learning for complex function approximation)

### 5.4 参数物理意义与选取指南 (Parameter Selection Guide)

为了方便实验，以下对关键参数的选取依据进行详细解释：

**1. DAC INL 系数 ($\gamma$)**
*   **代码默认值**: `0.01`
*   **物理含义**: 模拟 DAC 传输曲线的三次谐波失真分量。$\gamma=0.01$ 意味着在满量程输入时，由非线性引起的误差幅度约为信号幅度的 1% (即 -40dB 左右的谐波失真)。
*   **选取建议**:
    *   **0.01 (1%)**: 典型商用中低端 DAC 或未校准 DAC 的表现，适合测试模型的鲁棒性。
    *   **0.001 (0.1%)**: 高精度 DAC 的表现。
    *   **0.05 - 0.1**: 极端恶劣环境或低成本器件，用于压力测试。

**2. Rapp 模型平滑因子 ($p$)**
*   **代码默认值**: `3.0`
*   **物理含义**: 控制功放从线性区进入饱和区的 "膝部 (Knee)" 锐度。值越大，线性区越长，进入饱和越突然。
*   **选取建议**:
    *   **$p \approx 2.0$**: 典型的 GaAs (砷化镓) 功率放大器，过渡较平缓。
    *   **$p \approx 3.0$**: 典型的 GaN (氮化镓) 或 LDMOS 功率放大器，现代通信系统中最常用的经验值。
    *   **$p \to \infty$**: 理想软限幅器 (Ideal Clipper)。

**3. 饱和电压 ($V_{sat}$)**
*   **代码默认值**: `1.0`
*   **物理含义**: 功放的输出“天花板”。这并非为了主动限制 PAPR，而是模拟物理器件因供电电压有限而无法输出无限大信号的特性。当信号峰值超过此值时，会被压缩或截断，客观上确实会降低 PAPR，但也带来了非线性失真。
*   **选取建议**:
    *   **$V_{sat}=1.0$**: 标准配置。
    *   **$V_{sat} \to \infty$ (如 100.0)**: **模拟理想线性功放 (Ideal PA)**。此时相当于**不限制**，即仅有 DAC 量化噪声，没有任何 PA 饱和非线性。

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
# 使用 Rapp 模型训练 (推荐配置: p=3.0, sat=1.0)
# 此配置模拟 GaN 功放特性，并自动包含 DAC INL 非线性。
python train_resume.py --checkpoint_dir ./model/adda_rapp_snr10 \
    --resume_from ./model/baseline_snr10/model_epoch_400.pth \
    --enable_adda \
    --nonlinearity rapp \
    --adda_p 3.0 \
    --adda_sat 1.0 \
    --rho 1.0 --snr 10
```

**B. 使用 Tanh 模型 (Baseline)**
简单的双曲正切饱和模型。
公式：$$ V_{out} = \alpha \cdot \tanh(\beta \cdot V_{in}) $$
*   `--adda_alpha`: 幅度缩放因子。
*   `--adda_beta`: 输入增益因子。

```bash
# 使用 Tanh 模型训练 (alpha=1.0, beta=1.0)
python train_resume.py --checkpoint_dir ./model/adda_tanh_snr10 \
    --resume_from ./model/baseline_snr10/model_epoch_400.pth \
    --enable_adda \
    --nonlinearity tanh \
    --adda_alpha 1.0 \
    --adda_beta 1.0 \
    --rho 1.0 --snr 10
```

#### 步骤三：验证与测试 (Verification)
使用 `decompress.py` 进行端到端性能测试。由于我们内置了 `decompress.py` 会自动读取训练时的非线性配置（需手动指定参数以匹配训练设置），请确保测试参数与训练一致。

```bash
# 测试 Rapp 模型 (需与训练参数一致)
python decompress.py --checkpoint_dir ./model/adda_rapp_snr10 --model_name model_epoch_400.pth \
    --input_dir ./PointCloud-compression-geo/output/test \
    --output_dir ./PointCloud-compression-geo/decompressed/test_rapp \
    --nonlinearity rapp \
    --adda_p 3.0 \
    --adda_sat 1.0 \
    --snr 10

# 测试 Tanh 模型
python decompress.py --checkpoint_dir ./model/adda_tanh_snr10 --model_name model_epoch_400.pth \
    --input_dir ./PointCloud-compression-geo/output/test \
    --output_dir ./PointCloud-compression-geo/decompressed/test_tanh \
    --nonlinearity tanh \
    --adda_alpha 1.0 \
    --adda_beta 1.0 \
    --snr 10
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