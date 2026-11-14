---
type: entity
tags: [repo, vla, humanoid, nvidia, foundation-policy, lerobot, isaac-lab, deployment]
status: complete
updated: 2026-11-14
summary: "Isaac GR00T 是 NVIDIA 开源的人形 VLA 开发平台：GR00T N1.7 GA 参考代码、LeRobot 数据管线、后训练/评测/TensorRT 部署，并与 Isaac Lab-Arena、Teleop、Isaac ROS 组成端到端策略流水线。"
related:
  - ./nvidia-gr00t-e2e-g1-workflow.md
  - ../entities/paper-hrl-stack-34-gr00t_n1.md
  - ../entities/paper-gr00t-n1-5.md
  - ../entities/gr00t-wholebodycontrol.md
  - ../entities/lerobot.md
  - ../entities/isaac-gym-isaac-lab.md
  - ../entities/nvidia-gear-lab.md
  - ../concepts/foundation-policy.md
  - ../methods/vla.md
  - ../tasks/loco-manipulation.md
  - ./letools.md
  - ./paper-indi.md
  - ./perceptron-isaac-05.md
  - ./isaac-teleop.md
  - ./isaac-ros-deploy.md
  - ./nvidia-physical-ai-datasets.md
  - ./nvidia-gtc.md
original:
  - ../../sources/repos/isaac_gr00t.md
sources:
  - ../../sources/courses/nvidia_gr00t_e2e_g1_workflow.md
  - ../../sources/repos/isaac_gr00t.md
  - ../../sources/blogs/nvidia_develop_humanoid_robot_policies_isaac_gr00t.md
  - ../../sources/blogs/wechat_embodied_ai_lab_isaac_gr00t_n17_g1_e2e.md
  - ../../sources/papers/gr00t_n1_arxiv_2503_14734.md
---

# Isaac GR00T（人形 VLA 开发平台）

**Isaac GR00T** 是 NVIDIA 面向通用人形机器人的 **开源开发平台**：以 [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 仓库为工程中枢，托管 **GR00T N1.7**（GA）VLA 的权重、参考实现、LeRobot 数据格式与后训练/部署工具，并与 **Isaac Lab-Arena**、**Isaac Teleop**、**Isaac ROS + Jetson Thor** 组成可模块化拆用的端到端流水线。

> **名称消歧：** 本页是 **NVIDIA Isaac** 仿真/人形 VLA **平台**。Perceptron 2026-08 发布的开源通才模型也叫 Isaac，见 [Perceptron Isaac 0.5](./perceptron-isaac-05.md) — 二者无代码或权重关系。

> **论文机制深读：** [GR00T N1](../entities/paper-hrl-stack-34-gr00t_n1.md) — N1 双系统架构、数据金字塔与 GR-1 评测以 arXiv:2503.14734 为准。  
> **全身低层控制：** [GR00T-WholeBodyControl](../entities/gr00t-wholebodycontrol.md) — G1 上 `UNITREE_G1_SONIC` latent → SONIC WBC 解码。

## 一句话定义

Isaac GR00T 把「采集 demonstration → LeRobot 格式 → GR00T N1.7 后训练 → 仿真/真机闭环评测 → LEAPP/TensorRT 部署」收敛到 **单一开源栈**，让开发者从预训练 3B VLA 出发适配新 embodiment，而不必从零训练通才策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| VLM | Vision-Language Model | GR00T System 2 的多模态理解骨干 |
| DiT | Diffusion Transformer | flow-matching 动作头（连续动作去噪） |
| WBC | Whole-Body Control | 全身平衡与关节协调的低层控制器 |
| LEAPP | — | Isaac ROS 侧策略导出/部署 bundle 格式（GR00T 参考工作流） |
| GA | General Availability | N1.7 主分支的稳定正式发布状态 |
| EEF | End-Effector | 末端执行器；N1.7 默认相对 EEF delta 动作空间 |

## 为什么重要

- **填补「论文实体 ↔ 工程落地」缺口**：仓库已有 GR00T N1 论文页与 WBC 专仓，本页聚焦 **Isaac-GR00T 平台本身**——数据 schema、微调 CLI、Policy Server 与 benchmark 目录。
- **模块化端到端**：NVIDIA 技术博客强调可只用单组件（例如仅后训练脚本）或走完整五阶段栈，降低人形管线胶水成本。
- **LeRobot 生态对齐**：数据格式基于 LeRobot v2 + `modality.json`；HF LeRobot 亦提供 `groot` policy type，与本仓 reference 实现互补。
- **N1.7 GA 商用边界清晰**：代码 Apache 2.0；权重 NVIDIA Open Model License；Cosmos-Reason2-2B 为 gated HF 模型需单独申请访问。

## 核心原理

N1.7 的「模型内部」以仓库 `Gr00tN1d7Config`（`gr00t/configs/model/gr00t_n1d7.py`）为准，与 [N1 论文](./paper-hrl-stack-34-gr00t_n1.md)的双系统思路一脉相承，但主要超参已不同：

| 模块 | N1 论文（2B） | N1.7 代码默认值 |
|------|---------------|-----------------|
| VLM | Eagle-2（SmolLM2 + SigLIP-2），取第 12 层 | `nvidia/Cosmos-Reason2-2B`（Qwen3-VL），`select_layer=12`，`backbone_embedding_dim=2048`；默认 `tune_llm=False`、`tune_visual=False` |
| 动作头 | DiT + adaLN，交叉/自注意力交替 | flow-matching DiT，`num_layers=16`、32 头 × 48 维、`norm_type=ada_norm`、`interleave_self_attention=True`；`use_alternate_vl_dit=True`，每 2 个块（`attend_text_every_n_blocks=2`）切换关注文本 |
| 动作块 / 维度 | $H=16$ | `action_horizon=40`，`max_state_dim=max_action_dim=132`，`max_num_embodiments=32` |
| 去噪 | $K=4$，Beta(1.5,1) 采样 $\tau$，$s=0.999$ | `num_inference_timesteps=4`，`noise_beta_alpha=1.5`，`noise_beta_beta=1.0`，`noise_s=0.999`，1000 个时间步桶 |
| 状态处理 | per-embodiment MLP | 同族；另有 `state_dropout_prob=0.8`（模型配置默认；微调 CLI 默认 0.2）以降低对本体状态的依赖 |
| 动作表示 | 末端 axis-angle / 关节 | 微调入口强制 `use_relative_action=True`（相对 EEF delta） |

推理侧由 `Gr00tPolicy`（`gr00t/policy/gr00t_policy.py`）封装观测校验（`check_observation`）、动作校验（`check_action`）与 `Gr00tSimPolicyWrapper`；`run_gr00t_server.py` 把它包成 ZMQ `PolicyServer`（默认 `0.0.0.0:5555`，`strict=True`）；若只给 `--dataset-path` 则起 `ReplayPolicy`（必须同时给正的 `--execution-horizon`），用于不加载模型地验证环境。

## 平台五阶段（NVIDIA 栈）

```mermaid
flowchart LR
  A["① 仿真环境\nIsaac Lab-Arena"] --> B["② 数据采集\nIsaac Teleop"]
  B --> C["③ 格式转换\nHDF5 → LeRobot"]
  C --> D["④ 后训练\nIsaac-GR00T N1.7"]
  D --> E["⑤ 评测 / 部署\nArena 闭环 + Isaac ROS"]
```

| 阶段 | 典型产出 | 关键入口 |
|------|----------|----------|
| 仿真环境 | 场景 + 任务 + WBC 配置 | Isaac Lab-Arena asset registry |
| 数据采集 | HDF5 demonstrations | `record_demos.py`（OpenXR 等） |
| 格式转换 | `lerobot/` 目录（parquet + mp4 + meta） | `convert_hdf5_to_lerobot.py` + YAML 映射 |
| 公开数据 | [Physical AI HF 集合](./nvidia-physical-ai-datasets.md) | `GR00T-X-Embodiment-Sim`、Teleop 族、`LIBERO_LeRobot_v3` 等 |
| 后训练 | finetune checkpoint | `launch_finetune.py` + `--embodiment-tag` |
| 评测/部署 | success rate / LEAPP bundle | ZMQ Policy Server；Isaac ROS G1 教程 |

**工程提示：** 遥操作阶段选用的 **WBC**（如 AGILE WBC vs stand/walk 分解）会改变 joint-space 训练分布；应在采数前固定控制器，与目标部署栈一致。

## G1 仿真教程（Isaac Lab-Arena · 静态 pick-and-place）

NVIDIA 与 [具身智能研究室](https://mp.weixin.qq.com/s/Y2mlKtd-dGGdA33Sx_sDCw) 均以 **Unitree G1** 在货架前 **apple → plate** 任务示范完整链路（`galileo_g1_static_pick_and_place`）：

| 步骤 | 命令/产物 | 备注 |
|------|-----------|------|
| 搭环境 | Isaac Lab-Arena asset registry + `PickAndPlaceTask` | 静态任务用 **AGILE WBC**；完整代码见 [G1 仿真环境 GitLab Pages](https://unitree-g1-physical-ai-workflow-b42650.gitlab-master-pages.nvidia.com/simulation-workflow/sim-environment-code-review.html) |
| VR 采数 | `record_demos.py` + XR（Lab 3.x 走 [Isaac Teleop](./isaac-teleop.md)） | CloudXR + VR 头显；示例 400 条 HDF5（可多 session）。旧文档的 `--teleop_device openxr` 是 Lab 2.x 遗留路径 |
| 转 LeRobot | `convert_hdf5_to_lerobot.py` + `g1_static_apple_config.yaml` | 映射 `robot_joint_pos` / `processed_actions` / `robot_head_cam_rgb` |
| 后训练 | `launch_finetune.py` + `g1_sim_wbc_data_gr00t_n_1_7_config.py` | `--embodiment-tag new_embodiment`；Arena 外单独 checkout Isaac-GR00T |
| 闭环评测 | `policy_runner.py` + 远端 GR00T server | ZMQ；冒烟 `--num_steps 600`；统计 `--num_episodes 100` + `--num_envs 5` |

真机 G1 部署见 [Isaac ROS GR00T Reference Workflow](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html)（MCAP → LeRobot → LEAPP）；机载 ROS 2 运行时见 [Isaac ROS Deploy](./isaac-ros-deploy.md)。

## GR00T N1.7（当前 GA 主线）

| 字段 | 内容 |
|------|------|
| 基座权重 | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B)（约 3B 参数） |
| VLM 骨干 | [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)（Qwen3-VL 架构） |
| 动作头 | flow-matching **DiT**；`action_horizon` 40；state/action 维 132 |
| 动作空间 | **相对 EEF delta**（跨人/机 embodiment 共享，利于 EgoScale 人类视频预训练） |
| 许可 | 代码 Apache 2.0；权重 NVIDIA Open Model License |
| 历史分支 | [N1.6](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6) · [N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d5) |

相对 N1.6，N1.7 还强化：**多数据集 mixture 微调**、更广 benchmark 文档（RoboCasa、LIBERO、SimplerEnv、真机 G1）、**ONNX/TensorRT** 全链路导出，以及 server-client 推理硬化。

## 工程实践

| 项 | README / 源码给出的要点 |
|----|-------------------------|
| 硬件 | 推理 1 张 16 GB+ 显存 GPU（RTX 4090、L40、H100、Jetson Thor/Orin、DGX Spark）；微调建议 40 GB+，A6000 可行但更慢 |
| 吞吐（README 实测） | N1.7 在 RTX PRO 6000 Blackwell Server 约 **40 Hz**、Jetson AGX Thor 约 **28 Hz**（TensorRT，batch-1 LIBERO）；单卡训练步（batch 1，BF16，骨干冻结）H100 115.5 ms、L40 190.5 ms |
| 环境 | `uv sync --python 3.12`；视频后端仅 `torchcodec`（0.8.0 只支持 FFmpeg 4–7）；aarch64 平台须激活 `activate_<platform>.sh` 后用裸 `python`，不要 `uv run` |
| 门控 | 所有 checkpoint 首次加载会拉 gated 的 `nvidia/Cosmos-Reason2-2B`，需 HF 申请访问与登录，否则 `GatedRepoError` |
| 微调 | `launch_finetune.py`（tyro CLI）：`--base-model-path`、`--dataset-path`（多路径用 `os.pathsep` 分隔，`ds_weights_alpha` 控制混合权重）、`--embodiment-tag NEW_EMBODIMENT`、`--modality-config-path`；示例 `--max-steps 2000 --global-batch-size 32` |
| 调参经验 | README「Training Tips」：尽量加大 batch、训几千步；因图像增强非确定性，不同次运行可有 **5–6%** 方差；依赖本体状态的任务应调低 `--state_dropout_prob`（LIBERO-10 用 0.2，SimplerEnv Bridge 0.8、Fractal 0.5）；`--strict_vl_padding_mask` / `--strict_action_padding_mask` 默认保持旧 checkpoint 行为，新训练可开启（切换后须重导 TensorRT 引擎） |
| 评测 | 先 `open_loop_eval.py`（GT vs 预测 MSE 图），再 server-client 闭环；LIBERO、SimplerEnv、RoboCasa、SO100、DROID、RoboLab 各有 README |
| 许可 | 代码 Apache 2.0；权重 NVIDIA Open Model License |

## 仓库内五步（Isaac-GR00T README）

1. **Prepare data** — GR00T LeRobot 格式；内置 `demo_data/*` 可零配置试跑  
2. **Run inference** — 基座或 finetuned checkpoint + `--embodiment-tag`  
3. **Fine-tune** — `launch_finetune.py`（可 `--no-tune-llm`，调 visual/projector/diffusion）  
4. **Evaluate** — open-loop MSE 可视化 + closed-loop sim/真机  
5. **Deploy** — `run_gr00t_server.py` + `PolicyClient`；可选 TensorRT  

### 全身 G1（VLA + WBC 分层）

对 `UNITREE_G1_SONIC` embodiment，VLA 输出 **紧凑 latent action token**，由 [GEAR-SONIC](../methods/sonic-motion-tracking.md) 全身控制器解码为腿/臂/手关节命令——单策略可端到端协调 **操作 +  locomotion**（详见 [GR00T-WholeBodyControl](../entities/gr00t-wholebodycontrol.md) 教程）。

## 局限与风险

- **版本漂移：** 仓库 `main` 已是 N1.7，N1/N1.5 的行为只能从 `n1d5`、`n1d6` 分支或论文核对；本页的 N1.7 数值来自 README 与默认配置，**不等于**已复现的评测结论（README 也提示 5–6% 运行间方差）。
- **门控依赖：** Cosmos-Reason2-2B 为 gated 模型，离线/受限网络环境需提前准备权重。
- **平台环境脆弱：** README 记录了 FFmpeg 8 与 `torchcodec` 不兼容、CUDA 13 需要 Triton 补丁、GB300 (sm_103) 上 `torch.compile` 不可用等已知问题。
- **N1.7 的 20K 小时 EgoScale 人类视频与博客「32K/8K 小时」口径来自官方描述，我们未获取其数据集做核验。**（推测：具体混合比例未在 README 披露）

### 常见误区

1. **Isaac GR00T ≠ 仅 GR00T N1 论文** — 论文页描述 N1 研究贡献；本仓随版本演进至 **N1.7 GA**，架构与数据接口以 GitHub `main` 为准。  
2. **Embodiment tag 不可省略** — 决定 state/action/video 键与归一化；错 tag 会导致 silent 维度错配。  
3. **LeRobot 与本仓分工** — HF LeRobot 适合 `groot` 原生训练/rollout；**模型内部、部署脚本、benchmark 细节**以 Isaac-GR00T 为 reference。  
4. **WBC 与 VLA 仍分层验证** — 桌面短视界任务可能只需 AGILE WBC；loco-manip 需 SONIC 栈与 latent 解码联调。

## 关联页面

- 论文 canonical：[paper-hrl-stack-34-gr00t_n1.md](../entities/paper-hrl-stack-34-gr00t_n1.md)  
- 低层 WBC / SONIC：[gr00t-wholebodycontrol.md](../entities/gr00t-wholebodycontrol.md)  
- 数据互操作：[lerobot.md](../entities/lerobot.md)
- 官方数据索引：[nvidia-physical-ai-datasets.md](./nvidia-physical-ai-datasets.md) — GR00T / Teleop / LeRobot v3 子集
- 采数栈：[isaac-teleop.md](./isaac-teleop.md) — Lab 3.x XR 主线；MCAP / LeRobot；Televiz 把头显当瘦客户端  
- Kuavo 厂商胶水：[letools.md](./letools.md) — Learning 仓将 GR00T N1.5/N1.7 列入统一训练/client 部署  
- 仿真框架：[isaac-gym-isaac-lab.md](../entities/isaac-gym-isaac-lab.md)  
- 官方动手课：[nvidia-gr00t-e2e-g1-workflow.md](./nvidia-gr00t-e2e-g1-workflow.md) — G1 apple→plate **仿真/真机双路径** step-by-step 教程
- 全栈定位：[NVIDIA Physical AI 工具链技术地图](../overview/nvidia-physical-ai-toolchain-technology-map.md) — 本页五阶段在端到端七段里的上下游
- LEAPP 机载运行时：[isaac-ros-deploy.md](./isaac-ros-deploy.md) — Triton + `ros2_control` / topic 部署
- 概念层：[foundation-policy.md](../concepts/foundation-policy.md)、[vla.md](../methods/vla.md)  
- 跨本体 VLA 对照：[Robbyant（蚂蚁灵波）](./robbyant.md) — LingBot-VLA 2.0 论文把 GR00T N1.7 列为 generalist 对照基线
- 视觉 Sim2Real 姊妹仓：[gr00t-visual-sim2real.md](../entities/gr00t-visual-sim2real.md)
- 对象中心闭环增强：[POT-VLA](./paper-pot-vla.md) — 在 **GR00T-N1.7** 动作头插入持久 3D 对象 token + 几何谓词验收（G1 **39/80→71/80**；POT 侧未开源）
- 行为意图蒸馏：[Indi](./paper-indi.md) — 在 GR00T-N1.7 解码器中间层恢复局部目标（arXiv:2608.23478；未开源）
- 名称撞车：[Perceptron Isaac 0.5](./perceptron-isaac-05.md) — Perceptron 36B 稀疏 VLA；**不是** 本平台

- 仿真示范扩增：[Zero-HLM](./paper-zero-hlm.md) — 重建场景并以执行反馈生成 G1 移动操作训练数据。

## 参考来源

- **原文通读：** [isaac_gr00t.md](../../sources/repos/isaac_gr00t.md) — Isaac-GR00T 仓库 README 全文 + 关键源码（2026-11-07 复核）  
- [nvidia_gr00t_e2e_g1_workflow.md](../../sources/courses/nvidia_gr00t_e2e_g1_workflow.md) — NVIDIA Learning 官方 G1 E2E 动手课归档（2026-09-23）
- [nvidia_develop_humanoid_robot_policies_isaac_gr00t.md](../../sources/blogs/nvidia_develop_humanoid_robot_policies_isaac_gr00t.md) — NVIDIA Developer Blog 端到端平台介绍（2026-07-07）  
- [wechat_embodied_ai_lab_isaac_gr00t_n17_g1_e2e.md](../../sources/blogs/wechat_embodied_ai_lab_isaac_gr00t_n17_g1_e2e.md) — 具身智能研究室中文策展转载（G1 + VR/LeRobot 链路，2026-07-13）  
- [gr00t_n1_arxiv_2503_14734.md](../../sources/papers/gr00t_n1_arxiv_2503_14734.md) — GR00T N1 论文与白皮书  
- 代码：<https://github.com/NVIDIA/Isaac-GR00T>  
- 平台页：<https://developer.nvidia.com/isaac/gr00t>

## 推荐继续阅读

- [Develop Humanoid Robot Policies End-to-End with NVIDIA Isaac GR00T](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/) — 官方端到端博客  
- [NVIDIA GR00T G1 端到端参考 workflow](./nvidia-gr00t-e2e-g1-workflow.md) — 本库课程实体页；[官方站点](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html)  
- [GTC 2025 Humanoid Developer Day — Building Humanoid Robots](./nvidia-gtc.md) — Jim Fan 介绍 GR00T 四部分（On-Demand）
- [Isaac Teleop + GR00T 1.7 LeRobot 集成（HF Blog）](https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot)  
- [GR00T Reference Workflow for Unitree G1（Isaac ROS）](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html) — 真机 MCAP → LeRobot → LEAPP 部署  
- [LeRobot GR00T 文档](https://github.com/huggingface/lerobot/blob/main/docs/source/groot.mdx) — HF 侧 `groot` policy 工作流  
- [GR00T N1 论文阅读笔记（Robot_Learning_Paper_Notebooks）](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/GR00T_N1_Humanoid_Foundation_Model/GR00T_N1_Humanoid_Foundation_Model.html)
