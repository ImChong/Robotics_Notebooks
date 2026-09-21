# LoongForge

> 来源归档（GitHub 仓库深度复核）

- **标题：** LoongForge
- **类型：** repo
- **机构：** 百度智能云百舸（Baidu AI Cloud Baige）
- **链接：** https://github.com/baidu-baige/LoongForge
- **官网：** https://baidu-baige.github.io/LoongForge/
- **文档：** https://loongforge.readthedocs.io/en/latest/index.html
- **Docker Hub：** https://hub.docker.com/u/loongforge
- **许可：** Apache License 2.0
- **Stars：** ~575+（2026-09-21）
- **入库日期：** 2026-09-06（公众号全景初收录）；**深度复核：** 2026-09-21
- **一句话说明：** 百度百舸开源的高性能训练框架，覆盖 LLM / VLM / 扩散 / 具身（VLA & WAM）全模态；Megatron 栈 + torch-native 具身子系统，支持 NVIDIA GPU 与昆仑芯 XPU。
- **代码：** https://github.com/baidu-baige/LoongForge（**已开源**，Apache 2.0；v0.1.0 起官方 tag）
- **沉淀到 wiki：** [`wiki/entities/cn-os-loongforge.md`](../../wiki/entities/cn-os-loongforge.md)
- **前身产品：** [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr)（企业训练加速套件，最大生产规模 5000+ XPU）

---

## 核心定位

LoongForge 由百度智能云百舸团队开发，从 **AIAK-Training-LLM** 企业训练加速套件开源而来，目标是在 **保持 loss 曲线与基线对齐** 的前提下，显著降低 LLM、VLM、扩散与具身模型的训练成本。README 宣称相对开源基线最高 **5.04×** 吞吐（DeepSeek-V3.2 Lite 等），具身子系统对 DreamZero 等 WAM 可达 **4.38×**。

### 双栈架构

| 栈 | 适用模态 | 后端 | 并行策略 |
|----|----------|------|----------|
| **Megatron Stack** | LLM、VLM、扩散（Wan / Qwen-Image） | [Loong-Megatron](https://github.com/baidu-baige/Loong-Megatron)（patch 版 Megatron-LM） | TP / PP / EP / CP、MoE 负载均衡（TAOT）、异构并行 |
| **Torch-Native Stack**（`loongforge/embodied/`） | VLA、WAM（Pi0.5、GR00T、xVLA、DreamZero 等） | 纯 PyTorch DDP / ZeRO-1 / FSDP / HSDP | 与 Megatron core 解耦；数据侧含 LeRobot 格式后端 |

### 训练流水线（Workflow）

开箱即用：**Pretrain → MidTrain → SFT → LoRA**；内置数据集格式转换、sequence packing、Megatron ↔ HuggingFace 双向 checkpoint 转换与在线 HF 读写。

### 具身模型覆盖（2026-09）

Pi0.5、GR00T-N1.6/N1.7、xVLA、Wall-OSS-0.5、LingBot-VA、FastWAM、DreamZero、Cosmos3 等；统一评测模块覆盖 **LIBERO / CALVIN / SimplerEnv / RoboTwin**（Pi0.5 / xVLA / GR00T 等）。

### 硬件

- **NVIDIA GPU**：统一预构建 Docker 镜像（`hub.docker.com/u/loongforge`）
- **昆仑芯 XPU**：`examples_xpu/` 与独立安装教程（P800 等）

### 代表性优化

- **MoE**：EP 通信 overlap、TAOT 拓扑感知专家副本放置（arXiv:2608.03676，最高 74% 开销降低）
- **多模态**：ViT/LLM 异构并行、解耦 encoder-decoder 训练、DP 负载均衡
- **具身**：`torch.compile`、CUDA Graph、自定义 fused op、FP8 通信（FSDP2 Delta-FP8 AllGather / DDP grad all-reduce）

### 仓库布局（摘录）

```
LoongForge/
├── loongforge/train/          # pretrain / sft / diffusion 入口
├── loongforge/models/         # LLM / encoder / omni / diffusion 抽象
├── loongforge/embodied/       # VLA+WAM 独立子系统（train.py → trainer）
├── configs/models/            # Hydra YAML
├── examples/                  # GPU 启动脚本
├── examples_xpu/              # 昆仑芯脚本
├── third_party/Loong-Megatron/
└── tools/                     # checkpoint 转换、数据预处理
```

### 典型启动（具身 Pi0.5，摘自 docs）

```bash
# 见 examples/embodied/pi05/ 与 loongforge/embodied/train.py
python loongforge/embodied/train.py --config examples/embodied/pi05/...
```

---

## 开源状态

- **已开源**：GitHub 主仓、文档站、Docker 镜像、40+ 模型 example 脚本均可公开获取。
- **部分闭源**：README 提及部分高性能 CUDA fused op 仅在百舸平台提供；TileLang 版 FusedDSA 等已开源。

---

## 对 wiki 的映射

- [LoongForge](../../wiki/entities/cn-os-loongforge.md)
- 交叉：[LeRobot](../../wiki/entities/lerobot.md)（具身数据 `lerobot_dataset.py`）
- 交叉：[VLA 方法页](../../wiki/methods/vla.md)、[Isaac GR00T](../../wiki/entities/isaac-gr00t.md)（GR00T 训练加速示例）
