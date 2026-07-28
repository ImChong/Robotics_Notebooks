# RoboNaldo（OpenDriveLab/RoboNaldo）

> 来源归档

- **标题：** RoboNaldo — Accurate, Stable and Powerful Humanoid Soccer Shooting
- **类型：** repo（Isaac Lab 仿真训练）
- **来源：** OpenDriveLab / 香港大学 · 香港中文大学 · Archon Robotics
- **链接：** <https://github.com/OpenDriveLab/RoboNaldo>
- **克隆：** `https://github.com/OpenDriveLab/RoboNaldo.git`
- **配套部署仓：** <https://github.com/OpenDriveLab/RoboNaldo_Deploy>（子模块路径 `RoboNaldo_Deploy/`；归档见 [robonaldo-deploy.md](robonaldo-deploy.md)）
- **项目页：** <https://opendrivelab.com/RoboNaldo/>
- **论文：** <https://arxiv.org/abs/2606.11092>
- **许可：** MIT（根目录 `LICENCE`；GitHub SPDX：MIT）
- **入库日期：** 2026-07-28
- **一句话说明：** RoboNaldo **仿真训练**官方仓：BeyondMimic 风格 Isaac Lab 扩展 `whole_body_tracking` + RSL-RL PPO；右脚踢球 NPZ 与 Stage 1–3 YAML 预设；`train` / `play` / `eval` 与 ONNX 导出对接部署仓。
- **开源状态：** **已开源**（2026-06 News：Training and deployment code release）— 训练代码 + 默认 `motions/right_kick.npz`；G1 URDF 需另下；预训练权重走 W&B / 自训 checkpoint。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md`](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md)

## 仓库概况（2026-07-28 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`OpenDriveLab/RoboNaldo`） |
| 默认分支 | `main` |
| Stars / Forks | ~32 / ~3 |
| Topics | `humanoid-robotics`, `reinforcement-learning-environments`, `soccer-playing-robot` |
| 推荐基线 | Isaac Sim **5.1.0** · Isaac Lab **2.3.2** · Python **3.11** |
| Homepage | arXiv:2606.11092 |

## 为何值得保留

- **论文可复现入口：** 项目页 Code 链指向本仓；与 [RoboNaldo_Deploy](robonaldo-deploy.md) 构成 **训练 → ONNX → 真机/MuJoCo** 闭环。
- **三阶段课程落地：** README 把 Stage 1a/1b → 2a/2b → 3 映射到 `right_kick/*.yaml`，对齐论文 motion scaffold → 任意球 → 来球。
- **BeyondMimic 扩展自带：** 内含改造后的 `whole_body_tracking`，无需再装上游 BeyondMimic；包名冲突需校验 `importlib` 路径。

## README 入口（归纳）

| 组件 | 路径 / 命令 |
|------|-------------|
| 扩展安装 | `pip install -e source/whole_body_tracking` |
| G1 资产 | 下载 `unitree_description.tar.gz` → `source/.../assets/`（gitignore） |
| 默认运动 | `motions/right_kick.npz`（另有 CSV；可选 W&B upload） |
| 训练 | `scripts/rsl_rl/train.py --task Tracking-Body-Frame-Flat-G1-v0 --yaml right_kick/<preset>.yaml` |
| 回放 | `scripts/rsl_rl/play.py`（可导出 `exported/policy-obs.onnx`） |
| 评测 | `scripts/rsl_rl/eval.py` → `logs/rsl_rl/eval/` |
| 文档 | `docs/quickstart.md` · `docs/task_params.md` · `docs/rewards.md` · `README_CN.md` |

### 课程 YAML 映射

| Stage | 用途 | 预设 |
|-------|------|------|
| 1a | 平面 motion-tracking 先验 | `tracking_params.yaml` |
| 1b（可选） | 混合地形跟踪稳健 | `tracking_mixed_params.yaml` |
| 2a | 小范围静止球适应 | `task_params_1.yaml` |
| 2b | 更大范围任意球射门 | `task_params_2.yaml` |
| 3 | 来球 + jump trigger / 自适应采样 | `task_params_3.yaml` |

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体（主） | [`wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md`](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) |
| 任务背景 | [`wiki/tasks/humanoid-soccer.md`](../../wiki/tasks/humanoid-soccer.md) |
| 对照范式 | [`wiki/methods/paid-framework.md`](../../wiki/methods/paid-framework.md) |
| Stage 1 跟踪 | [`wiki/methods/beyondmimic.md`](../../wiki/methods/beyondmimic.md) |
| 项目页 | [`sources/sites/opendrivelab-robonaldo.md`](../sites/opendrivelab-robonaldo.md) |
| 部署仓 | [`robonaldo-deploy.md`](robonaldo-deploy.md) |

## 参考链接

- 训练仓：<https://github.com/OpenDriveLab/RoboNaldo>
- 部署仓：<https://github.com/OpenDriveLab/RoboNaldo_Deploy>
- 项目页：<https://opendrivelab.com/RoboNaldo/>
- 论文：<https://arxiv.org/abs/2606.11092>
- 上游致谢：Isaac Lab · [BeyondMimic / whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) · RSL-RL
