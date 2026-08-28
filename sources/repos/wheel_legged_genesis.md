# wheel_legged_genesis

> 来源归档

- **标题：** wheel_legged_genesis（Genesis 双轮足 RL）
- **类型：** repo
- **来源：** Albusgive（个人 / 社区）
- **链接：** https://github.com/Albusgive/wheel_legged_genesis
- **星标（截至 2026-08-28）：** 368
- **最近推送：** 2025-07-10
- **主要语言：** Python
- **许可证：** MIT
- **分类：** 强化学习训练 / Genesis / 轮腿双足 / sim2sim
- **入库日期：** 2026-08-28
- **一句话说明：** 在 Genesis 上用 vendored RSL-RL 训练双轮足（CJ-003 wheelfoot / pointfoot），含域随机、课程与自定义地形，策略可迁到 MuJoCo（Python 或 C++）。
- **沉淀到 wiki：** 是 → [`wiki/entities/wheel-legged-genesis.md`](../../wiki/entities/wheel-legged-genesis.md)
- **机构：** 无独立机构条目（个人维护）
- **项目页：** 无独立 `*.github.io`；演示在 B 站（README 列出 4 条视频）

---

## README 要点（编译自上游）

- 定位：**Reinforcement learning of wheel-legged robots based on Genesis**。
- 系统：Ubuntu 20.04/22.04/24.04，Python ≥ 3.10；NVIDIA / AMD GPU 或 CPU。README **明确禁止** `pip install genesis==0.2.1`，须本地安装 Genesis **main**（API 已变）。
- 安装：`pdm install`，或手动装 Genesis + tensorboard / pygame / opencv + `cd rsl_rl && pip install -e .`。
- 入口：
  - 训练：`locomotion/wheel_legged_train.py`（`pdm run` 或 `python`）
  - 评估：`locomotion/wheel_legged_eval.py`（默认实验名 `wheel-legged-walking`，导出 JIT `policy.pt`）
  - 点足变体：`point_foot_loc/point_foot_*.py`
- 机型资产：`assets/mjcf/CJ-003/`（`CJ-003-wheelfoot.xml` / `CJ-003-pointfoot.xml`）与对应 URDF；**8 动作**：左右 hip/thigh/calf + 左右轮。关节 PD `kp=30, kv=0.8`，轮 `kv=1.0`；控制周期 100 Hz，`simulate_action_latency=True`。
- 地形：`agent_train_gym`（粗糙路 + 连续坡）、`agent_eval_gym`（精简评测）、`circular`；自定义见 `assets/terrain`。
- 手柄 / 键盘：线速度、偏航、腿长、单腿控制、重置；演示技能「太空步」「铁山靠」。
- sim2sim：`sim2sim/` 提供 **Python `gs2mj.py`** 与 **C++ `gs2mj`** 两条迁到 MuJoCo 的路径；`scence.xml` 需改绝对路径。
- Changelog 到 v0.0.7（声称 sim2sim 已能稳定过 MuJoCo）。TODO 仍开放：外力干扰、高速控制。
- 建议：NVIDIA 用 Genesis `gpu`/`cuda` backend；AMD 用 `vulkan`。

## 开源状态

- **已开源**：公开 GitHub 仓库（MIT）；仓内含训练/评估脚本、MJCF/URDF、vendored `rsl_rl`、MuJoCo sim2sim。
- **真机部署不在本仓**：README 主线是 Genesis 训练 + MuJoCo sim2sim；无官方 ROS / 板载推理包。

## 对 wiki 的映射

- 实体页：[`wiki/entities/wheel-legged-genesis.md`](../../wiki/entities/wheel-legged-genesis.md)
- 形态概念：[`wiki/concepts/wheel-legged-biped.md`](../../wiki/concepts/wheel-legged-biped.md)
- 仿真器：[`wiki/entities/genesis-sim.md`](../../wiki/entities/genesis-sim.md)
- 训练核：[`wiki/entities/rsl-rl.md`](../../wiki/entities/rsl-rl.md)
