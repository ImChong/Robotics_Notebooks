---
type: entity
tags: [repo, reinforcement-learning, genesis, locomotion, wheel-legged, rsl-rl, sim2sim, independent-maintainer]
status: complete
updated: 2026-08-28
related:
  - ./genesis-sim.md
  - ./rsl-rl.md
  - ../concepts/wheel-legged-biped.md
  - ./tita-rl.md
  - ./isaac-rl-two-wheel-legged-bot.md
  - ./mujoco.md
  - ../tasks/hybrid-locomotion.md
  - ../tasks/locomotion.md
  - ../concepts/domain-randomization.md
  - ../concepts/curriculum-learning.md
sources:
  - ../../sources/repos/wheel_legged_genesis.md
summary: "Albusgive/wheel_legged_genesis：在 Genesis 上用 RSL-RL 训练 CJ-003 双轮足（轮足/点足），含课程与自定义地形，策略可 JIT 导出并迁到 MuJoCo sim2sim。"
---

# wheel_legged_genesis

**wheel_legged_genesis** 是社区仓库 [`Albusgive/wheel_legged_genesis`](https://github.com/Albusgive/wheel_legged_genesis)：在 [Genesis](./genesis-sim.md) 上做 **双轮足** locomotion RL，训练核是仓内 vendored 的 [RSL-RL](./rsl-rl.md)。

## 一句话定义

用 Genesis 并行仿真 + PPO 训练 CJ-003（8 关节：腿 PD + 轮速），再把 JIT 策略搬进 MuJoCo 做 sim2sim；适合不想上 Isaac 全家桶、又要轮腿双足的实验。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Genesis | Genesis Embodied AI 仿真器 | 本仓的训练仿真后端 |
| RSL-RL | Robotic Systems Lab RL | `OnPolicyRunner` + PPO，仓内 `rsl_rl/` |
| MJCF | MuJoCo XML | `assets/mjcf/CJ-003/` 与 sim2sim 场景 |
| JIT | Just-In-Time | `wheel_legged_eval.py` 把 actor 存成 `policy.pt` |
| DR | Domain Randomization | v0.0.2 起加入；与课程、自定义地形一起开 |

## 为什么重要

- **Genesis 上少见的轮腿完整示例**：官方 Genesis 教程偏通用刚体；本仓把地形 PNG、课程、手柄遥操和 sim2sim 收成一条可跑流水线。
- **轮足 / 点足对照**：同一 CJ-003 家族提供 wheelfoot 与 pointfoot 两套 MJCF，方便看「有没有轮」对奖励与终止的影响。
- **迁移路径短**：评估脚本直接 `torch.jit.script`；MuJoCo 侧有 Python 与 C++ 两条 `gs2mj`。

## 核心原理

| 项 | 内容 |
|----|------|
| 维护者 | 独立维护者 Albusgive（非 Genesis AI 公司仓） |
| 动作 | 8 DoF：`hip/thigh/calf` ×2 + `wheel` ×2；轮与腿用不同 PD/`kv` |
| 频率 | `dt=0.01`（100 Hz），`simulate_action_latency=True` |
| 终止 | 俯仰/滚转超阈（训练期更宽以免 episode 过短）或指定 link 触地 |
| 地形 | `agent_train_gym` / `agent_eval_gym` / `circular`；height field 来自 PNG |
| 交互 | 手柄/键盘改 \(v,\omega\)、腿长、单腿、重置；演示「太空步」「铁山靠」 |

```mermaid
flowchart LR
  A[CJ-003 URDF/MJCF] --> B[Genesis WheelLeggedEnv]
  B --> C[RSL-RL OnPolicyRunner]
  C --> D[logs/model_*.pt]
  D --> E[JIT policy.pt]
  E --> F[MuJoCo gs2mj.py / C++]
```

## 工程实践

1. **不要** `pip install genesis-world==0.2.1`；按 README 从 Genesis **main** 本地安装（API 已变）。NVIDIA 用 `gpu`/`cuda` backend，AMD 用 `vulkan`。
2. 建议 `pdm install`；或手动 `pip install -e rsl_rl` 后跑 `python locomotion/wheel_legged_train.py`。
3. 评估：`python locomotion/wheel_legged_eval.py`（默认 `--exp_name wheel-legged-walking`）；点足走 `point_foot_loc/`。
4. sim2sim：改 `sim2sim/scence.xml` 里的绝对路径；Python 为 `python gs2mj.py`，C++ 需 libtorch + 系统 MuJoCo。
5. 优先用 **release tag**，README 写明 `main` 不一定稳。入库时最新叙述到 **v0.0.7**。

## 局限与风险

- **开源状态：已开源**（MIT）；训练、评估、sim2sim 入口齐全。
- **无官方真机包**：停在 MuJoCo；外力干扰与高速控制仍在 TODO。
- **最后推送 2025-07-10**：Genesis API 继续变时要自己跟；不要假设 pip 轮子能直接装。
- **个人维护**：与 [tita_rl](./tita-rl.md) / Flamingo 的厂商或实验室栈相比，硬件口径和 sim2real 证据更弱。

## 关联页面

- [轮腿双足](../concepts/wheel-legged-biped.md)
- [Genesis](./genesis-sim.md)
- [RSL-RL](./rsl-rl.md)
- [MuJoCo](./mujoco.md)
- [tita_rl](./tita-rl.md)
- [Isaac-RL-Two-wheel-Legged-Bot](./isaac-rl-two-wheel-legged-bot.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Curriculum Learning](../concepts/curriculum-learning.md)
- [Hybrid Locomotion](../tasks/hybrid-locomotion.md)

## 参考来源

- [sources/repos/wheel_legged_genesis.md](../../sources/repos/wheel_legged_genesis.md)
- 上游：<https://github.com/Albusgive/wheel_legged_genesis>

## 推荐继续阅读

- Genesis 仓库：<https://github.com/Genesis-Embodied-AI/Genesis>
- 作者 B 站演示（README 第一条）：<https://www.bilibili.com/video/BV14eKKeiEJB/>
