# Humanoid-GPT（GalaxyGeneralRobotics 官方仓库）

- **标题：** Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking
- **类型：** repo
- **仓库：** <https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>
- **论文：** <https://arxiv.org/abs/2606.03985>
- **项目页：** <https://qizekun.github.io/Humanoid-GPT/>
- **收录日期：** 2026-06-04
- **最近复核：** 2026-06-19
- **Stars / Forks：** ~243 / 0（2026-06-19 检索）
- **许可证：** Apache 2.0
- **维护方：** Galaxy General Robotics（Galbot Inc. 关联开源组织）

## 一句话摘要

CVPR 2026 论文 **Humanoid-GPT** 的官方代码入口：已发布 **推理 / 评测 / 真机部署** 与 **ONNX 预训练 checkpoint**；**训练代码与训练数据** 仍标注为 TODO。仓库实现 **HME、GQS、因果 Transformer tracking policy** 等侧模块，并以 **Unitree G1（29-DoF）** 为主平台。

## 为何值得保留

- **可复现锚点**：与 arXiv / 项目页并列，便于跟进工程细节（仿真、checkpoint、部署脚本、数据格式）。
- **生态位**：与 [SONIC](../../wiki/methods/sonic-motion-tracking.md)（NVIDIA，100M MLP）形成 **Transformer + 2B 帧** 对照的开源跟踪基线候选。
- **工程可用性（2026-06-19）**：README 标明 **推理与部署代码**、**`storage/ckpts/pns_wo_priv216.onnx`** 已发布；真机 / Jetson / BrainCo 手部分支有独立 `DEPLOY.md`。

## 发布状态（README TODO 区块）

| 组件 | 状态 |
|------|------|
| 推理与部署代码 | ✅ 已发布 |
| 预训练 checkpoint（ONNX） | ✅ `storage/ckpts/pns_wo_priv216.onnx` |
| 训练代码 | ⏳ 计划中 |
| 训练数据 | ⏳ 计划中 |

## 环境与安装

- **Python 3.12** + Conda；`pip install -e ".[cuda]"`（CUDA 12.x GPU）；MacOS 可 `.[cpu]` 或仅真机部署 `.`。
- 仿真加速：**MuJoCo-MJX**；Mac 测试可用 `mjpython` 跳过 `jax[cuda12]`。
- **G1 硬件版本**：环境变量 `G1_VERSION`（默认 `5010`），资产路径 `storage/assets/unitree_g1_${G1_VERSION}/`。

## 推理与评测（`scripts/`）

```bash
python -m scripts.app                    # Gradio 交互 demo
python -m scripts.inference --load_path storage/ckpts/pns_wo_priv216.onnx --mocap_path storage/test
python -m scripts.eval_parallel --load_path storage/ckpts/pns_wo_priv216.onnx --mocap_path storage/test --workers 32 --privileged
python -m scripts.vis --mocap_path storage/test
```

- **动作格式**：`.npz` 含 `qpos`，或 `root_pos` / `root_rot` / `dof_pos`。
- **重定向后处理**：`tracking/convert_qpos2kpt.py`（单文件）、`tracking/convert_parallel.py`（批量）将 retarget 结果转为 policy 消费的 keypoint 表示。

## 真机部署（`deploy/`）

- 总览：[`deploy/DEPLOY.md`](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/main/deploy/DEPLOY.md)
- 仿真：`python -m deploy.play_track --track-dir storage/test`
- 真机：`python -m deploy.play_track --real --net <nic_name>`
- **机载（Jetson Orin）**：`deploy/onboard_deploy/`（[`DEPLOY_ONBOARD.md`](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/main/deploy/onboard_deploy/DEPLOY_ONBOARD.md)）
- **机载无 GMR**：`onboard_deploy_wo_GMR/` — 主机侧 retarget 流式输入
- **BrainCo 灵巧手**：`deploy/brainco/`（[`BRAINCO.md`](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/main/deploy/brainco/BRAINCO.md)）

## 仓库结构（README 摘要）

| 目录 | 职责 |
|------|------|
| `tracking/` | 推理核心：常量、`infer_utils`、`policy.py`（ONNX wrapper）、keypoint 转换与 tracking 指标 |
| `scripts/` | `inference` / `eval_parallel` / `vis` / `app`（Gradio） |
| `deploy/` | 真机部署子模块（见上） |
| `projects/hme/` | Harmonic Motion Encoder（Periodic Autoencoder） |
| `projects/gqs/` | General Quality Selection（物理 + 多样性评分） |
| `projects/tracking_transformer/` | Transformer tracking policy（推理 / 部署） |
| `utils/` | MuJoCo / MJX 仿真、变换、视频渲染 |
| `storage/` | 资产、配置、样例轨迹、已发布 checkpoint |

## 架构要点（README 对齐）

- **因果 Transformer + RoPE**（Rotary Position Embeddings），支持变长运动序列。
- 预训练 **2B 帧** 统一 mocap 语料；**零样本** 跟踪未见动作，无需微调。
- 平台：**Unitree G1** 全身 29-DoF；GPU 加速仿真 MuJoCo-MJX。

## 对 Wiki 的映射

- [Humanoid-GPT（论文实体页）](../../wiki/entities/paper-humanoid-gpt.md)
- [SONIC（规模化运动跟踪）](../../wiki/methods/sonic-motion-tracking.md) — 同任务不同规模/结构路线的对照阅读

## 参考来源（原始）

- GitHub 仓库 README — <https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>（2026-06-19 检索）
