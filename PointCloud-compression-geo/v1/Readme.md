# 点云轻量化联合编码调制系统 (JCMPC)

本项目实现了基于论文 **"Lightweight Joint Coding-Modulation Optical Fiber Communication System for Point Cloud"** 的点云联合信源信道编码（JSCC）方案。
项目核心在于通过 **3D 卷积神经网络** 直接将点云体素块映射为模拟符号，并针对 **ADDA（模数/数模转换）** 引入的物理损伤（量化噪声、INL/DNL、功放非线性）进行了精细建模和补偿。

---

## 1. 项目简介 (Project Overview)

*   **核心特性**: 端到端联合优化，抗 "悬崖效应"。
*   **网络结构**: 双分支对称卷积神经网络 (Dual-branch Symmetric CNN)。
*   **ADDA 优化**: 内置可配置的数模转换损伤模型，支持 **纯 ADC/DAC 精度优化** 和 **功放非线性补偿** 双模式。

---

## 2. 快速开始 (Quick Start)

本项目提供了一键式自动化脚本，可自动完成 **压缩 -> 解压 -> 合并** 的全流程实验。

### 推荐方式：运行自动化管线
请直接运行 `run_pipeline.py`，所有配置（ADDA 模式、参数、路径）均可在脚本顶部的 `CONFIG` 区域修改。

```bash
# 确保在项目根目录下
python PointCloud-compression-geo/g_train32_49152_ch48_downsample16_snr10_norelu/run_pipeline.py
```

在 `run_pipeline.py` 中，你可以轻松配置：
```python
# --- ADDA 配置 ---
DISABLE_ADDA = False   # 默认开启 ADDA 补偿
NONLINEARITY = "none"  # "none" (纯 ADC 模式) 或 "rapp" (功放模式)
DNL_SIGMA    = 0.0     # DNL 强度 (如 0.05)
INL_GAMMA    = 0.01    # INL 强度 (如 0.01)
```

---

## 3. ADDA 模块配置详解 (Configuration)

本项目采用了高度可配置的 ADDA 通道模型，支持两种主要工作模式。**默认情况下，ADDA 模块是开启的**。

### 模式 A: 纯 ADC/DAC 优化 (默认模式)
**适用场景**: 仅关注数模/模数转换的 **量化精度**、**INL (积分非线性)** 和 **DNL (差分非线性)**。物理通道的功率放大器 (PA) 被视为理想线性。

*   **激活方式**: 不指定 `--nonlinearity` 参数 (默认为 `none`)。
*   **信号链路**: `Input -> [DNL Noise] -> [Quantization (DAC)] -> [INL Distortion] -> [Ideal PA] -> [AWGN] -> Output`
*   **关键参数**:
    *   `--adda_bits`: 量化位数 (默认 8)。
    *   `--dnl_sigma`: **(新增)** 差分非线性 (DNL) 强度，模拟量化阈值抖动。默认 `0.0` (关闭)。建议值 `0.01-0.05`。
    *   `--inl_gamma`: **(新增)** 积分非线性 (INL) 强度，模拟传输曲线弯曲。默认 `0.01`。

### 模式 B: 功放非线性仿真 (Legacy Mode)
**适用场景**: 除了 ADC/DAC 损伤外，还需要模拟真实物理 **功率放大器 (PA)** 的饱和非线性 (如 Rapp 模型)。

*   **激活方式**: 指定 `--nonlinearity rapp`。
*   **信号链路**: `Input -> ... -> [INL] -> [Rapp Power Amplifier] -> [AWGN] -> Output`
*   **关键参数**:
    *   `--nonlinearity rapp`: 启用 Rapp 模型。
    *   `--adda_p`: 平滑因子 (默认 3.0)。模拟 GaN (p~3) 或 GaAs (p~2) 功放。
    *   `--adda_sat`: 饱和电压 (默认 1.0)。

### 如何关闭 ADDA?
如果需要回退到纯数字信道训练（仅 AWGN，无任何硬件损伤模拟）：
*   请添加参数 `--disable_adda`。

---

## 4. 手动运行指南 (Manual Usage)

如果你需要单独运行训练、压缩或解压步骤，请参考以下命令。

### 4.1 训练 (Training)

#### (1) 基准模型训练 (Baseline)
```bash
python train_resume.py --checkpoint_dir ./model/baseline --disable_adda
```

#### (2) ADDA 适应性微调 (Adaptation)
在基准模型基础上，开启 ADDA 损伤进行微调。

**只有 ADC/DAC 损伤 (模式 A):**
```bash
python train_resume.py \
    --checkpoint_dir ./model/adda_pure_adc \
    --resume_from ./model/baseline/model_epoch_400.pth \
    --dnl_sigma 0.05 \
    --inl_gamma 0.01
```

**包含功放非线性 (模式 B):**
```bash
python train_resume.py \
    --checkpoint_dir ./model/adda_rapp \
    --resume_from ./model/baseline/model_epoch_400.pth \
    --nonlinearity rapp \
    --adda_p 3.0
```

### 4.2 压缩 (Compression)
使用训练好的模型对点云进行编码。
```bash
python compress.py \
    --input_dir ./data/test/blocks \
    --output_dir ./output/compressed \
    --checkpoint_dir ./model/adda_pure_adc \
    --model_name model_epoch_500.pth \
    --dnl_sigma 0.05 --inl_gamma 0.01  # 保持与训练一致
```

### 4.3 解压 (Decompression)
对压缩数据进行重建。
```bash
python decompress.py \
    --input_dir ./output/compressed \
    --output_dir ./output/decompressed \
    --checkpoint_dir ./model/adda_pure_adc \
    --model_name model_epoch_500.pth \
    --dnl_sigma 0.05 --inl_gamma 0.01
```

---

## 5. 参数物理意义速查

| 参数名 | 默认值 | 物理含义 | 选取建议 |
| :--- | :--- | :--- | :--- |
| **`--adda_bits`** | 8 | DAC/ADC 的量化精度。 | 8-bit (标准), 10-bit (高精), 4-6 bit (低功耗) |
| **`--dnl_sigma`** | 0.0 | **差分非线性 (DNL)**。模拟量化台阶宽度的随机抖动。 | `0.0`: 理想；`0.01-0.05`: 高精度仪表；`>0.1`: 劣质器件 |
| **`--inl_gamma`** | 0.01 | **积分非线性 (INL)**。模拟传输曲线的整体弯曲(三次谐波)。 | `0.01`: 典型商用 DAC (1% 失真)；`0.001`: 高端仪器 |
| **`--adda_p`** | 3.0 | (Rapp模型) **膝部系数**。控制功放进入饱和的急剧程度。 | `2.0`: GaAs; `3.0`: GaN/LDMOS; `inf`: 理想硬限幅 |
| **`--adda_sat`** | 1.0 | (Rapp模型) **饱和电压**。功放最大输出幅值。 | `1.0`: 归一化标准值；设为很大(如100)可近似线性功放 |

---

## 6. 网络架构版本 (Model Versions)

本项目提供两种网络架构版本，可通过 `--model_version` 参数切换。

### TAE (v1) - 基线架构
*   **文件**: `TAE.py`
*   **结构**: 双分支残差块 + ReLU
*   **特点**: 参数量小，训练快

### TAE_v2 (v2) - 改进架构 (NEW)
*   **文件**: `TAE_v2.py`
*   **改进点**:
    | 改进 | 说明 |
    |---|---|
    | **SE 注意力块** | Squeeze-and-Excitation，自适应通道加权 |
    | **Group Norm** | 批大小无关的归一化，训练更稳定 |
    | **GELU 激活** | 平滑梯度，保留负值信息 |
    | **Pre-Act ResBlock** | GN→GELU→Conv，更好的梯度流 |
*   **预期提升**: PSNR +0.5~1.5 dB

### 使用方式
```bash
# 使用基线架构 (默认)
python train_resume.py --model_version v1 ...

# 使用改进架构
python train_resume.py --model_version v2 ...
```