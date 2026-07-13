# NVIDIA Isaac-GR00T — 原始资料归档

- **来源**：https://github.com/NVIDIA/Isaac-GR00T
- **类型**：repo
- **机构**：NVIDIA（GEAR / Isaac）
- **归档日期**：2026-07-13
- **GitHub**：约 7.5k+ stars（2026-07）；**代码** Apache-2.0，**模型权重** NVIDIA Open Model License
- **关联技术博客**：[Develop Humanoid Robot Policies End-to-End with NVIDIA Isaac GR00T](../blogs/nvidia_develop_humanoid_robot_policies_isaac_gr00t.md)（NVIDIA Developer Blog，2026-07-07）

## 一句话说明

**Isaac-GR00T** 是 NVIDIA 开源的 **GR00T N1.7 VLA** 参考实现与开发平台：提供 LeRobot 格式数据管线、`launch_finetune.py` 后训练、ZMQ Policy Server、ONNX/TensorRT 部署，并与 Isaac Lab-Arena / Isaac Teleop / Isaac ROS 组成端到端人形策略开发工作流。

## 为什么值得保留

- **N1.7 General Availability（GA）**：主分支为稳定 GA 发布；Cosmos-Reason2-2B（Qwen3-VL）骨干 + flow-matching DiT 动作头；Apache 2.0 商用友好
- **端到端工程闭环**：README 明确五步——准备数据 → 推理 → 微调 → 评测 → 部署；含 demo 数据集、benchmark 示例与 `Gr00tPolicy` API
- **LeRobot 互操作**：GR00T LeRobot 格式（v2 + `meta/modality.json`）；LeRobot 侧 `groot` policy type 与本仓 reference 实现分工清晰
- **全身人形路径**：`UNITREE_G1_SONIC` embodiment + [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) 解码 latent action → 全身关节命令
- **多平台部署**：x86 dGPU、Jetson Thor/Orin、DGX Spark；`uv` 环境管理 + 分平台 `install_deps.sh`

## 核心摘录（README + 官方博客归纳）

### 1）GR00T N1.7 相对 N1.6 的关键变化

| 维度 | N1.7 | N1.6（对照） |
|------|------|--------------|
| VLM 骨干 | `nvidia/Cosmos-Reason2-2B`（Qwen3-VL） | Eagle-Block2A-2B-v2 |
| 模型包 | `gr00t_n1d7` | `gr00t_n1d6` |
| 状态/动作维 | 132 / action horizon 40 | 29 / horizon 16 |
| 动作空间 | **相对 EEF（delta pose）**，跨人/机 embodiment 共享 | 绝对/混合（版本差异） |
| 预训练数据 | 约 20K 小时 EgoScale 人类视频 + 多样机器人演示（博客称 ~32K 小时真人演示 + ~8K 小时仿真） | 以机器人数据为主 |
| 部署 | 全链路 ONNX + TensorRT | 部分路径 |

### 2）仓库内五步工作流

1. **Prepare data** — 遥操作/仿真采集 → GR00T LeRobot 格式（`meta/modality.json` 描述 state/action/video 键）
2. **Run inference** — 基座 `nvidia/GR00T-N1.7-3B` 零样本或 finetuned checkpoint
3. **Fine-tune** — `gr00t/experiment/launch_finetune.py`；支持多数据集 mixture、`--embodiment-tag`
4. **Evaluate** — open-loop（`open_loop_eval.py`）+ closed-loop（server-client ZMQ）
5. **Deploy** — `run_gr00t_server.py` + TensorRT；真机经 Isaac ROS LEAPP 导出（见博客 G1 参考工作流）

### 3）主要 checkpoint 与 embodiment tag

| Checkpoint | 类型 | Embodiment Tag |
|------------|------|----------------|
| `nvidia/GR00T-N1.7-3B` | Base（3B） | 见 pretrain tags |
| `nvidia/GR00T-N1.7-LIBERO` | Finetuned | `LIBERO_PANDA` |
| `nvidia/GR00T-N1.7-DROID` | Finetuned | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` |
| `nvidia/GR00T-N1.7-SimplerEnv-*` | Finetuned | Bridge / Fractal 对应 tag |
| — | 全身 G1 | `UNITREE_G1_SONIC`（latent → SONIC WBC） |

### 4）NVIDIA 栈五阶段（技术博客 Table 1）

| 阶段 | NVIDIA 组件 | 功能 |
|------|-------------|------|
| 仿真环境 | Isaac Lab-Arena | 场景组合、任务定义、策略训练/测试环境 |
| 数据采集 | Isaac Teleop（CloudXR VR） | 高质量遥操作 demonstration |
| 策略训练 | Isaac GR00T 1.7 + 训练脚本 | 仿真+真机 demo 后训练 VLA |
| 策略评测 | Isaac Lab-Arena | 闭环仿真评测（server-client） |
| 真机部署 | Isaac ROS + Jetson Thor | LEAPP bundle 导出与边缘推理 |

### 5）工程注意事项

- **git-lfs** 必需（`demo_data/` parquet）
- **Hugging Face gated model**：Cosmos-Reason2-2B 需申请访问并 `hf auth login`
- **FFmpeg 4–7**：`torchcodec` 视频后端；Ubuntu 新发行版 FFmpeg 8 不兼容
- **Fine-tune 硬件**：推荐 40GB+ VRAM（H100/L40）；推理 16GB+ 即可
- **LeRobot v3 → v2**：`scripts/lerobot_conversion/convert_v3_to_v2.py`

## 对 wiki 的映射

1. **[Isaac GR00T（开发平台）](../../wiki/entities/isaac-gr00t.md)** — 本仓作为平台/engineering 实体页主来源
2. **[GR00T N1（论文实体）](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)** — arXiv:2503.14734 机制与 N1 论文评测；代码入口即本仓
3. **[GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md)** — G1 全身 latent 解码与 SONIC WBC
4. **[LeRobot](../../wiki/entities/lerobot.md)** — 数据格式与 `groot` policy 互操作
5. **[Foundation Policy / VLA](../../wiki/concepts/foundation-policy.md)** — 通才基础策略概念层

## 引用（仓库 README）

```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and others},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```
