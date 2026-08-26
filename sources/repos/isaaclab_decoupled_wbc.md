# IsaacLab-Decoupled-WBC — 原始资料归档

- **来源**：https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC
- **类型**：repo
- **机构**：卡内基梅隆大学 / 德克萨斯大学阿灵顿分校 / 博世人工智能中心（HTD 论文作者单位）
- **归档日期**：2026-08-26
- **Stars**：约 26（2026-08-26）
- **许可：** BSD-3-Clause（派生自 [LeggedLab](https://github.com/Hellod035/LeggedLab)；`dataset/g1/CMU/` 动捕子集**不**覆盖该软件许可）
- **一句话说明：** Isaac Lab 上的轻量解耦全身/下肢控制器：PPO teacher → BC+DAgger student → JIT 真机部署；单 GPU 可跑完全训练管线，是 HTD 在 Unitree G1 上的稳定控制底座。

## 为什么值得保留

- HTD 论文里的 **Lower-Body Controller / Decoupled WBC** 首次有可运行官方代码与 example checkpoint
- 命令接口同时跟踪 **速度（vx, vy, yaw rate）** 与 **躯干位姿（height, roll, pitch, yaw）**，覆盖蹲下、大幅俯仰等极端姿势
- 维护路径明确：student 控 **15 个下肢+腰** 关节，上肢在真机保持默认位姿；训练时可用 AMASS 重定向臂运动作干扰
- 浏览器 MuJoCo Demo 与 Conda / Docker 双安装，降低复现门槛

## 仓库结构（2026-08-26）

| 组件 | 路径 | 职责 |
|------|------|------|
| Teacher 训练 | `legged_lab/scripts/train_teacher.py`、`scripts/train_teacher.sh` | PPO，默认 `g1_flat`，12288 env / 250k iter |
| Student 蒸馏 | `legged_lab/scripts/train_student.py`、`scripts/train_student.sh` | 一键 BC 250k → DAgger 至 600k |
| 仿真 Play | `play_teacher.py` / `play_student.py` | 键盘驱动速度与躯干命令 |
| 真机部署 | `deploy/deploy_student_htd.py`、`deploy/configs/g1_student_htd.yaml` | Unitree SDK2 + JIT，50 Hz |
| 示例权重 | `example/`、`deploy/policy/g1_student/` | teacher `model_000.pt` + student JIT |
| 环境/奖励 | `legged_lab/envs/g1/g1_config.py` | 命令范围、per-axis curriculum、跟踪奖励 |
| 臂运动回放 | `dataset/g1/CMU/`、`legged_lab/utils/amass.py` | 训练时重放 14 个臂关节；部署不用 |
| 安装 | `scripts/setup_conda.sh`、`scripts/setup_docker.sh` | Isaac Sim 5.0 + Lab 2.2.0，或镜像 `chrisyrniu/htd-wbc:isaaclab-2.2.0` |

## 开源边界

- **已发布：** teacher/student 训练与评测、example checkpoint、G1 真机部署、浏览器 Demo。
- **不在本仓：** 全身 VR 遥操作、触觉采数、HTD 策略训练（见 [humanoid-touch-dream](./humanoid_touch_dream.md) checklist，截至 2026-08-26 仍 on-going）。
- **动捕数据许可：** `dataset/g1/CMU/` 选自 [AMASS Retargeted for G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)，受原数据集条款约束。

## 关键复现路径

1. `bash scripts/setup_conda.sh` 或 `bash scripts/setup_docker.sh`
2. `bash scripts/play_student.sh` — 不训练即可验证 bundled JIT
3. （可选）`bash scripts/train_teacher.sh` → `bash scripts/train_student.sh`
4. `cd deploy && python deploy_student_htd.py --config_path configs/g1_student_htd.yaml`

文档：[`docs/training.md`](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC/blob/main/docs/training.md)、[`docs/deployment.md`](https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC/blob/main/docs/deployment.md)。

## 交叉链接

| 档案 | 关系 |
|------|------|
| [humanoid_touch_dream.md](./humanoid_touch_dream.md) | HTD 论文仓；本仓作为 submodule `htd_wbc/isaaclab_decoupled_wbc` |
| [humanoid-touch-dream.md](../sites/humanoid-touch-dream.md) | 项目页与 Demo |
| [humanoid_touch_dream.md](../papers/humanoid_touch_dream.md) | 论文摘录 |

## 对 wiki 的映射

- [HTD 解耦 WBC（实体）](../../wiki/entities/htd-decoupled-wbc.md)
- [HTD 方法页](../../wiki/methods/humanoid-transformer-touch-dreaming.md)
- [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
- [Isaac Lab](../../wiki/entities/isaac-lab.md)
- [Unitree G1](../../wiki/entities/unitree-g1.md)
