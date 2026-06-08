# holosoma（Amazon FAR 人形 RL + 重定向框架）

> 来源归档

- **标题：** holosoma
- **类型：** repo
- **来源：** Amazon FAR（Frontier AI & Robotics）
- **链接：** <https://github.com/amazon-far/holosoma>
- **License：** Apache-2.0
- **Stars：** ~1.4k（2026-06，以 GitHub 为准）
- **入库日期：** 2026-06-08
- **一句话说明：** Amazon FAR 开源的 **人形全身 RL 训练与部署** 框架（希腊语「全身」），集成 **locomotion / whole-body tracking** 与 **OmniRetarget 运动重定向**；支持 IsaacGym、IsaacSim、MJWarp、MuJoCo 与 G1/T1 真机推理管线。
- **沉淀到 wiki：** [`wiki/entities/holosoma.md`](../../wiki/entities/holosoma.md)、[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)

---

## 核心定位

**Holosoma** 是 OmniRetarget 论文承诺的 **公开代码入口**：将训练、推理与重定向拆为三个顶层包，共享配置与 Wandb 实验管理。

```
src/
├── holosoma/              # 核心训练（locomotion & whole-body tracking）
├── holosoma_inference/    # 仿真/真机推理与部署
└── holosoma_retargeting/  # 人形 MoCap → 机器人轨迹（OmniRetarget 引擎）
```

---

## 功能摘要（README）

| 维度 | 内容 |
|------|------|
| 仿真器 | IsaacGym、IsaacSim、MuJoCo Warp（MJWarp）、MuJoCo（仅推理） |
| 算法 | PPO、FastSAC |
| 机器人 | Unitree G1、Booster T1 |
| 任务 | 速度跟踪 locomotion、whole-body tracking（WBT） |
| 部署 | sim-to-sim / sim-to-real 共享推理管线 |
| 重定向 | 保留物体/地形交互的人形 MoCap 重定向（OmniRetarget） |
| 日志 | Wandb 视频、ONNX 自动上传、从 Wandb 直接加载 checkpoint |

---

## 快速上手（官方脚本）

| 场景 | 命令 |
|------|------|
| IsaacGym 环境 | `bash scripts/setup_isaacgym.sh` |
| IsaacSim 环境 | `bash scripts/setup_isaacsim.sh`（Ubuntu 22.04+） |
| MJWarp + MuJoCo | `bash scripts/setup_mujoco.sh` 或 `setup_mujoco_via_uv.sh` |
| 推理/部署 | `bash scripts/setup_inference.sh` |
| 重定向 | `bash scripts/setup_retargeting.sh` |

**训练示例（G1 + FastSAC + IsaacGym）：**

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

**端到端 Demo：**

```bash
# OMOMO 数据：重定向 + WBT 策略训练
bash demo_scripts/demo_omomo_wb_tracking.sh

# LAFAN1 数据：重定向 + WBT 策略训练
bash demo_scripts/demo_lafan_wb_tracking.sh
```

---

## 与 OmniRetarget 的关系

- **论文实现载体：** OmniRetarget 的 interaction-mesh 重定向与下游 **5 reward + 4 DR** 的 WBT 训练均在此仓库发布。
- **数据配套：** 公开重定向轨迹见 HuggingFace [`omniretarget/OmniRetarget_Dataset`](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)；LAFAN1 重定向因许可 **不随数据集发布**，需用本仓库 `holosoma_retargeting` 自行重定向。
- **下游引用：** [PHP](../papers/php_parkour_arxiv_2602_15827.md) 等 Amazon FAR 系工作将 OmniRetarget 作为原子技能重定向上游。

---

## 文档入口（仓库内）

- 训练：[`src/holosoma/README.md`](https://github.com/amazon-far/holosoma/blob/main/src/holosoma/README.md)
- 推理/部署：[`src/holosoma_inference/README.md`](https://github.com/amazon-far/holosoma/blob/main/src/holosoma_inference/README.md)
- 重定向：[`src/holosoma_retargeting/holosoma_retargeting/README.md`](https://github.com/amazon-far/holosoma/blob/main/src/holosoma_retargeting/holosoma_retargeting/README.md)

## 对 wiki 的映射

- 实体页：[`wiki/entities/holosoma.md`](../../wiki/entities/holosoma.md)
- 论文实体：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)
- 问题域：[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)
