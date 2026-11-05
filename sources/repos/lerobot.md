# LeRobot

> 来源归档

- **标题：** LeRobot
- **类型：** repo
- **链接：** https://github.com/huggingface/lerobot
- **Hub 组织页：** https://huggingface.co/lerobot
- **文档：** https://huggingface.co/docs/lerobot/index
- **RTC 文档：** https://huggingface.co/docs/lerobot/rtc — [`lerobot-rtc-docs.md`](../sites/lerobot-rtc-docs.md)
- **EnvHub 文档：** https://huggingface.co/docs/lerobot/envhub
- **LeRobotDataset v3 文档：** https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- **Stars：** ~24k+（2026-09）
- **入库日期：** 2026-05-27
- **最近复核：** 2026-09-06（EnvHub / `lerobot-eval`）
- **一句话说明：** Hugging Face 具身智能全栈：PyTorch 库与 CLI（GitHub）+ Hub 上的模型 / LeRobotDataset / **EnvHub 仿真环境**。
- **代码：** https://github.com/huggingface/lerobot（**已开源**，Apache 2.0）
- **沉淀到 wiki：** [lerobot](../../wiki/entities/lerobot.md)、[lerobot-humanoid](../../wiki/entities/lerobot-humanoid.md)、[lerobot-envhub](../../wiki/concepts/lerobot-envhub.md)、[lerobot-dataset-v3](../../wiki/concepts/lerobot-dataset-v3.md)
- **LeRobot Humanoid 四仓：** [lerobot_humanoid_hardware.md](lerobot_humanoid_hardware.md)、[lerobot_humanoid_runtime.md](lerobot_humanoid_runtime.md)、[lerobot_legged_zoo.md](lerobot_legged_zoo.md)、[lerobot_humanoid_identification.md](lerobot_humanoid_identification.md)
- **交叉归档：** [lerobot-huggingface-org.md](../sites/lerobot-huggingface-org.md)、[lerobot-envhub-docs.md](../sites/lerobot-envhub-docs.md)、[lerobot-dataset-v3-docs.md](../sites/lerobot-dataset-v3-docs.md)

---

## 核心定位

Hugging Face 具身智能全栈：**GitHub** 侧为 Python 包、`lerobot-record` / `lerobot-train` / **`lerobot-eval`** CLI 与硬件驱动；**Hub** 侧托管预训练模型、LeRobot 格式数据集、**EnvHub 仿真环境 Git 仓**、Collections 与 Spaces。

### 三层 Hub 资产（2026-09）

| 层 | 入口 | 说明 |
|----|------|------|
| **Models** | `huggingface.co/lerobot` | π0、SmolVLA、ACT 等 checkpoint |
| **Datasets** | LeRobotDataset v3 | Parquet + MP4；`LeRobotDataset("lerobot/...")` |
| **Environments (EnvHub)** | `make_env("org/repo")` | Hub 仓 `env.py` + `make_env`；见 [EnvHub 文档](../sites/lerobot-envhub-docs.md) |

### 评测 CLI（README）

```bash
# 内置 benchmark
lerobot-eval --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero --env.task=libero_object --eval.n_episodes=10

# Hub 环境（需 trust_remote_code）
lerobot-eval --env.type=isaaclab_arena --env.hub_path=nvidia/isaaclab-arena-envs \
  --policy.path=nvidia/smolvla-arena-gr1-microwave --trust_remote_code=True
```

**EnvHub 契约：** Hub 仓暴露 `make_env(n_envs, use_async_envs, cfg?)` → `VectorEnv` / `Env` / 多任务 `dict`；加载器在 `lerobot.envs`。

### 策略族（README 表，摘录）

Imitation：ACT、Diffusion、VQ-BeT；RL：HIL-SERL、TDMPC；VLA：π0/π0.5、GR00T N1.7、SmolVLA、XVLA、EVO1 等；World Model：VLA-JEPA、FastWAM 等。

### 硬件与插件

原生：SO100、LeKiwi、Koch、HopeJR、Reachy2、OpenARM、Unitree G1、reBot B601 等。第三方通过 `lerobot_robot_*` / `lerobot_teleoperator_*` / `lerobot_camera_*` 包名自动发现。

本条目亦为 [导航·SLAM·自动驾驶栈 21 仓索引](navigation_slam_autonomy_stack_catalog.md) 组成部分。

---

## 对 wiki 的映射

- [LeRobot](../../wiki/entities/lerobot.md)
- [LeRobot EnvHub](../../wiki/concepts/lerobot-envhub.md)
- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- 总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

## README 模型清单（2026-11-05 快照）

核对范围：[README 固定版本](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/README.md#sota-models)。以下逐项保留上游分类与官方文档入口；不代表本库已完成每个模型的全文及源码核查。

| 类别 | 模型 | 官方文档 |
|------|------|----------|
| 模仿学习 | ACT | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_act_README.md) |
| 模仿学习 | Diffusion | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_diffusion_README.md) |
| 模仿学习 | VQ-BeT | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_vqbet_README.md) |
| 模仿学习 | Multitask DiT Policy | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_multi_task_dit_README.md) |
| 强化学习 | HIL-SERL | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/hilserl.mdx) |
| 强化学习 | TDMPC | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_tdmpc_README.md) |
| 强化学习 | QC-FQL | README 标注 coming soon；未提供文档链接 |
| 视觉语言动作模型 | Pi0 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/pi0.mdx) |
| 视觉语言动作模型 | Pi0Fast | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/pi0fast.mdx) |
| 视觉语言动作模型 | Pi0.5 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/pi05.mdx) |
| 视觉语言动作模型 | OpenGalaxea G0.5 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/g05.mdx) |
| 视觉语言动作模型 | FineART-VLA | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/fineart_vla.mdx) |
| 视觉语言动作模型 | GR00T N1.7 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_groot_README.md) |
| 视觉语言动作模型 | SmolVLA | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/policy_smolvla_README.md) |
| 视觉语言动作模型 | XVLA | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/xvla.mdx) |
| 视觉语言动作模型 | EO-1 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/eo1.mdx) |
| 视觉语言动作模型 | MolmoAct2 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/molmoact2.mdx) |
| 视觉语言动作模型 | LingBot-VLA 2.0 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/lingbot_vla_v2.mdx) |
| 视觉语言动作模型 | WALL-OSS | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/walloss.mdx) |
| 视觉语言动作模型 | Wall-OSS-0.5 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/wall_oss_05.mdx) |
| 视觉语言动作模型 | EVO1 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/evo1.mdx) |
| 视觉语言动作模型 | DM05 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/dm05.mdx) |
| 世界模型 | VLA-JEPA | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/vla_jepa.mdx) |
| 世界模型 | LingBot-VA | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/lingbot_va.mdx) |
| 世界模型 | FastWAM | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/fastwam.mdx) |
| 世界模型 | Long-WAM | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/longwam.mdx) |
| 世界模型 | LaWAM | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/lawam.mdx) |
| 世界模型 | FLUX 3 Action | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/flux3.mdx) |
| 世界模型 | Cosmos3 | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/cosmos3.mdx) |
| 奖励模型 | SARM | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/sarm.mdx) |
| 奖励模型 | TOPReward | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/topreward.mdx) |
| 奖励模型 | Robometer | [上游文档](https://github.com/huggingface/lerobot/blob/9e5ee9859480e425a97a6836bb3fee479539beb5/docs/source/robometer.mdx) |

## 模型 → 详情节点映射（2026-11-05 核对完成）

32 个模型逐个回到 LeRobot 固定提交的文档与 `src/lerobot/policies/` 源码，并核对原论文；每个模型一个独立节点（QC-FQL 上游无文档与代码，不建节点）。各节点均含 `## 流程总览` 与 `## 源码运行时序图`（LeRobot 版），并记录 LeRobot 实现与论文的差异。

| 类别 | 模型 | 详情节点 |
|------|------|----------|
| 模仿学习 | ACT | [paper-act](../../wiki/entities/paper-act.md) |
| 模仿学习 | Diffusion | [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md) |
| 模仿学习 | VQ-BeT | [paper-vq-bet-behavior-generation-latent-actions](../../wiki/entities/paper-vq-bet-behavior-generation-latent-actions.md) |
| 模仿学习 | Multitask DiT Policy | [multitask-dit-policy](../../wiki/entities/multitask-dit-policy.md) |
| 强化学习 | HIL-SERL | [paper-rcl-2410-21845-precise-and-dexterous-robotic-manipulation-via-h](../../wiki/entities/paper-rcl-2410-21845-precise-and-dexterous-robotic-manipulation-via-h.md) |
| 强化学习 | TDMPC | [paper-tdmpc-temporal-difference-model-predictive-control](../../wiki/entities/paper-tdmpc-temporal-difference-model-predictive-control.md) |
| 强化学习 | QC-FQL | 无（上游仅 README 一行 coming soon，无文档/代码） |
| VLA | Pi0 | [paper-pi0](../../wiki/entities/paper-pi0.md) |
| VLA | Pi0Fast | [paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la](../../wiki/entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md) |
| VLA | Pi0.5 | [paper-pi05-open-world-vla](../../wiki/entities/paper-pi05-open-world-vla.md) |
| VLA | OpenGalaxea G0.5 | [paper-galaxea-g05](../../wiki/entities/paper-galaxea-g05.md) |
| VLA | FineART-VLA | [paper-fineart-vla](../../wiki/entities/paper-fineart-vla.md) |
| VLA | GR00T N1.7 | [paper-gr00t-n1-7](../../wiki/entities/paper-gr00t-n1-7.md) |
| VLA | SmolVLA | [paper-smolvla](../../wiki/entities/paper-smolvla.md) |
| VLA | XVLA | [paper-x-vla](../../wiki/entities/paper-x-vla.md) |
| VLA | EO-1 | [paper-eo1](../../wiki/entities/paper-eo1.md) |
| VLA | MolmoAct2 | [paper-molmoact2](../../wiki/entities/paper-molmoact2.md) |
| VLA | LingBot-VLA 2.0 | [lingbot-vla-v2](../../wiki/entities/lingbot-vla-v2.md) |
| VLA | WALL-OSS | [cn-os-wall-x](../../wiki/entities/cn-os-wall-x.md) |
| VLA | Wall-OSS-0.5 | [paper-wall-oss-0-5](../../wiki/entities/paper-wall-oss-0-5.md) |
| VLA | EVO1 | [paper-evo1-lightweight-vla](../../wiki/entities/paper-evo1-lightweight-vla.md) |
| VLA | DM05 | [dexmal-dm05](../../wiki/entities/dexmal-dm05.md) |
| 世界模型 | VLA-JEPA | [paper-sa-2602-10098-vla-jepa-enhancing-vision-language-action-model](../../wiki/entities/paper-sa-2602-10098-vla-jepa-enhancing-vision-language-action-model.md) |
| 世界模型 | LingBot-VA | [paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g](../../wiki/entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md) |
| 世界模型 | FastWAM | [paper-fast-wam](../../wiki/entities/paper-fast-wam.md) |
| 世界模型 | Long-WAM | [paper-long-wam-scaling-context](../../wiki/entities/paper-long-wam-scaling-context.md) |
| 世界模型 | LaWAM | [paper-lawam](../../wiki/entities/paper-lawam.md) |
| 世界模型 | FLUX 3 Action | [flux-3-action](../../wiki/entities/flux-3-action.md) |
| 世界模型 | Cosmos3 | [cosmos-3](../../wiki/entities/cosmos-3.md) |
| 奖励模型 | SARM | [paper-sarm](../../wiki/entities/paper-sarm.md) |
| 奖励模型 | TOPReward | [paper-topreward](../../wiki/entities/paper-topreward.md) |
| 奖励模型 | Robometer | [paper-robometer](../../wiki/entities/paper-robometer.md) |

**需注意的身份细节：** Pi0Fast 复用 FAST 论文节点（2501.09747）；GR00T N1.7 无独立论文，仅有 HF 模型卡；Multitask DiT 无独立论文（参考 TRI LBM 2507.05331 与博文）；DM05 未找到 DM0.5 论文，仅官方博客；TDMPC 的 LeRobot 实现源码文件头写明实际是 FOWM（2310.16029）而非原版 TD-MPC；WALL-OSS 一代与 Wall-OSS-0.5 在 LeRobot 中是不同策略类型。QC-FQL 的状态待上游发布后复核。
