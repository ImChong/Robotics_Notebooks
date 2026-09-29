# carm-lerobot

> 来源归档

- **标题：** carm-lerobot（LeRobot revised MAXHUB）
- **类型：** repo
- **机构：** 视源股份（CVTE Robotics / MAXHUB）
- **链接：** https://github.com/cvte-robotics/carm-lerobot
- **星标（截至 2026-09-29）：** ~2
- **最近推送：** 2026-09-28
- **主要语言：** Python
- **许可证：** Apache-2.0（与上游 LeRobot 一致）
- **包版本（pyproject）：** lerobot **0.5.1**（本仓为 fork 全量树，非子模块）
- **分类：** 模仿学习 · 桌面/协作臂
- **入库日期：** 2026-09-29
- **一句话说明：** 视源 CARM 机械臂官方 LeRobot 改版：内嵌 LeRobot 0.5.1，扩展 A3/D3 真机与 `a3_leader` 手柄遥操作，一条 CLI 走采集–回放–ACT/Diffusion/SmolVLA/π0.5/WALL-OSS 训练与真机推理。
- **沉淀到 wiki：** 是 → [`wiki/entities/carm-lerobot.md`](../../wiki/entities/carm-lerobot.md)
- **依赖 SDK：** [`sources/repos/pycarm.md`](pycarm.md)（`pip install carm`）
- **组织地图：** [`sources/repos/cvte_robotics.md`](cvte_robotics.md)

---

## README 要点（编译自上游，2026-09-29）

- **定位：** README 标题为「lerobot revised MAXHUB」；面向 **CARM A3** 协作臂（及仓内 **D3** 驱动），在 Hugging Face LeRobot 之上接 **真机 SDK** 与 **网页端手柄遥操作**。
- **环境：** Miniforge + Python 3.12；`pip install -e .` 安装本 fork；另需 `pip install carm`（PyPI/SDK，见 pycarm）。
- **硬件类型（CLI）：**
  - `--robot.type=a3_follower` — 从臂 / 录数与推理执行端
  - `--teleop.type=a3_leader` — 主臂 / 手柄遥操作（网页连接手柄）
  - `--robot.addr='{"right":"10.42.0.101"}'` — 单臂或双臂 IP 映射；双臂时 `left`/`right` 各一台
- **`move_mode`：** `joint` / `pose` / `both` — 决定 episode 存关节、末端位姿或二者；采集时 `robot` 与 `teleop` 的 `move_mode` 必须一致；`both` 可用 `examples/extract_joint_and_pose.py` 拆成两份数据集。
- **`enable_action`：** 采数时为 **false**（只录不驱）；推理时为 **true**（默认 true）。
- **相机：** OpenCV `index_or_path`；名称（如 `left`/`hand`/`right`）须与训练集一致；支持手动白平衡/曝光。
- **数据默认路径：** `~/.cache/huggingface/lerobot/`；可选 `--dataset.push_to_hub`。
- **策略（README 示例）：** ACT、Diffusion Policy、SmolVLA、π0.5（`lerobot/pi05_base`）、WALL-OSS（`wall_x` + `x-square-robot/wall-oss-flow`）。
- **仿真侧：** 仓内 `examples/carm-mujoco/` 含 A3 MJCF 与 tutorial 采数/推理脚本；独立组织仓 [`carm-mujoco`](https://github.com/cvte-robotics/carm-mujoco) 亦存在。

## 代码布局（与复现相关的路径）

| 路径 | 职责 |
|------|------|
| `src/lerobot/robots/carm_a3/` | A3 从臂 `Robot` 实现（`carm.CArmSingleCol`，单/双臂） |
| `src/lerobot/robots/carm_d3/` | D3 机型驱动 |
| `src/lerobot/teleoperators/a3_leader/` | 手柄 Leader 遥操作 |
| `examples/extract_joint_and_pose.py` | `move_mode=both` 数据集拆分 |
| `examples/carm-mujoco/` | MuJoCo 模型与 tutorial |

## 开源状态

- **已开源：** 公开 GitHub 仓库 `cvte-robotics/carm-lerobot`（Apache-2.0）；训练/采数/推理 CLI 与 A3 驱动源码均在仓内。
- **外部依赖：** 真机控制依赖 **`carm` Python 包**（[`pycarm`](https://github.com/cvte-robotics/pycarm)）；无单独项目页，以 GitHub README 为复现入口。
- **未在本仓发布：** 预训练 checkpoint 与 HF 数据集需用户自建；π0.5 / WALL-OSS 示例从公开 Hub 基座微调。

## 对 wiki 的映射

- 实体页：[`wiki/entities/carm-lerobot.md`](../../wiki/entities/carm-lerobot.md)
- 框架枢纽：[`wiki/entities/lerobot.md`](../../wiki/entities/lerobot.md)
- 任务交叉：[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)、[`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)
