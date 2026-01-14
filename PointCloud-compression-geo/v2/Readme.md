# 点云轻量化联合编码调制系统 V2 (JCMPC-V2)

本项目实现了基于 **3D 卷积神经网络** 的点云联合信源信道编码（JSCC）方案，支持 **ADDA（模数/数模转换）** 损伤的端到端补偿。

> **V2 核心改进**: 使用 **可学习的 INL 模型 (DeepINL)** 替代固定三次失真，自适应补偿硬件非线性。

---

## 1. 核心特性

| 特性 | 说明 |
|---|---|
| **端到端优化** | 抗 "悬崖效应" |
| **可学习 INL** | 残差 MLP 自适应学习硬件失真 |
| **模块化 ADDA** | 量化 → INL → [可选 PA] → AWGN |

---

## 2. 快速开始

```bash
# 从项目根目录运行
python PointCloud-compression-geo/v2/run_pipeline.py
```

在 `run_pipeline.py` 中配置：
```python
DISABLE_ADDA = False      # 开启 ADDA
NONLINEARITY = "none"     # "none" (纯 ADC) 或 "rapp" (功放)
INL_HIDDEN_DIM = 64       # DeepINL 隐藏层维度
```

---

## 3. ADDA 信号链路

```
Input → [Quantization] → [Learnable INL: DeepINL] → [Optional PA] → [AWGN] → Output
```

> **注意**: V2 版本移除了 DNL 噪声，INL 由神经网络 (`DeepINL`) 自适应学习。

---

## 4. DeepINL 模块

```python
class DeepINL(nn.Module):
    """可学习的 INL 失真模型 (残差 MLP)"""
    def __init__(self, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # 残差连接: output = input + learned_deviation
        return x + self.net(x.view(-1, 1)).view(x.shape)
```

**优点**:
- 自适应拟合任意形状的 INL 曲线
- Tanh 激活函数确保平滑输出
- 残差连接保证收敛稳定

---

## 5. 训练示例

```bash
# 使用可学习 INL 训练
python train_resume.py \
    --checkpoint_dir ./model/learnable_inl \
    --resume_from ./model/baseline/model_epoch_400.pth \
    --inl_hidden_dim 64
```

---

## 6. 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--adda_bits` | 8 | 量化精度 |
| `--inl_hidden_dim` | 64 | DeepINL 隐藏层维度 |
| `--nonlinearity` | none | `none` / `rapp` / `tanh` |
| `--adda_p` | 3.0 | Rapp 模型平滑因子 |
| `--adda_sat` | 1.0 | Rapp 模型饱和电压 |

---

## 7. 与 V1 的区别

| 项目 | V1 | V2 |
|---|---|---|
| **INL 模型** | 固定三次失真 (`x + γ·x³`) | 可学习 MLP (`DeepINL`) |
| **DNL 噪声** | 支持 (`--dnl_sigma`) | 移除 |
| **自适应性** | 手动调参 | 端到端学习 |
| **参数** | `--dnl_sigma`, `--inl_gamma` | `--inl_hidden_dim` |