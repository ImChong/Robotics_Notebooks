# CoRe（Contact-Aware Motion Retargeting）

> 来源归档

- **标题：** CoRe: Contact-Aware Motion Retargeting
- **类型：** repo
- **链接：** https://github.com/tmjeong1103/CoRe
- **发布：** [v0.1.0](https://github.com/tmjeong1103/CoRe/releases/tag/v0.1.0)（2026-08-12）
- **许可：** 代码 Apache-2.0；示例动作 CC BY 4.0；捆绑机器人描述保留厂商许可（见 `docs/licenses.md`）
- **入库日期：** 2026-08-15
- **一句话说明：** 高丽大学 Robot Intelligence Lab 开源的接触感知全身重定向工具：把 Kimodo `.npz` / GEM-X `.pt` 的 SOMA 人体运动，经 DMR + 接触精炼映射到 11 台捆绑人形，导出无 pickle 的 `core-robot-motion-v1` `.npz`。
- **沉淀到 wiki：** 是 → [`wiki/entities/core-retarget.md`](../../wiki/entities/core-retarget.md)

## 开源核查（步骤 2.5）

- **代码：** **已开源、可运行** — <https://github.com/tmjeong1103/CoRe>（CLI `core-retarget`、Python `Retargeter`、`core-retarget serve` 本地/HF 浏览器演示）。
- **网页体验：** <https://huggingface.co/spaces/robotaemoon/CoRe>
- **论文项目页：** [CoRe-page](https://tmjeong1103.github.io/CoRe-page/)（Humanoids 2025）、[RMR](https://tmjeong1103.github.io/RMR/)（IROS 2025）
- **边界：** v0.1.0 覆盖 **重定向 + 接触精炼 + 预览导出**。Humanoids 论文中的 **text-to-motion** 与 **contact-aware RL 训练** 未随本仓发布。
- **验证（v0.1.0 发布说明）：** macOS 371 tests / 348 subtests；Ubuntu CI 覆盖 Python 3.10–3.13、native backend、无头渲染与 Docker Space。

## 摘录要点

### 输入 / 输出

| 侧 | 说明 |
|----|------|
| 输入 `.npz` | [Kimodo](https://github.com/nv-tlabs/kimodo) 已评估 SOMA77 全局关节位置/旋转 |
| 输入 `.pt` | [GEM-X](https://github.com/NVlabs/GEM-X) SOMA body parameters + static-contact logits（需 `gemx` extra；须显式 FPS，捆绑示例 30 Hz） |
| 输出 | `core-robot-motion-v1`：时间戳、MuJoCo `qpos`、命名关节布局、接触信息；`allow_pickle=False` |

扩展名即适配器契约：两者先归一到同一不可变 SOMA77，再进 DMR。

### 九段制品边界（`docs/architecture.md`）

1. `1_contacts.npz` — 源容器校验 + 提供方感知足接触
2. `2_dmr.npz` — 源身体目标 → 选定机器人（RMR 方向向量重定向）
3. `3_initial_collision.npz` — 初始手臂自碰精炼
4. `4_target_trajectories.npz` — 根 / 踝 / 足底 / 趾轨迹
5. `5_ara.npz` — 根轨迹与接地偏置
6. `6_fpa_targets.npz` — 接触感知落脚目标
7. `7_fpa_ik.npz` — 落脚 IK 与接地
8. `8_final.npz` — 最终手臂自碰精炼
9. `9_diagnostics.npz` — 终态诊断后写出 `final/robot_motion.npz`

### 捆绑 11 机（`docs/robots.md`）

| ID | 厂商 | 机型 | 驱动 DoF |
|----|------|------|----------|
| `g1` | Unitree | G1 29-DOF | 29 |
| `h1` | Unitree | H1 | 20 |
| `h2` | Unitree | H2 | 31 |
| `r1` | Unitree | R1 | 29 |
| `k1` | ROBOTIS | K1 | 23 |
| `apollo` | Apptronik | Apollo | 32 |
| `oli` | LimX Dynamics | Oli | 31 |
| `n1` | Fourier | N1 | 23 |
| `adam` | PNDbotics | ADAM Lite | 25 |
| `t1` | Booster | T1 | 23 |
| `pm01` | ENGINEAI | PM01 | 24 |

切换机型只需 `--robot <id>`。部分厂商模型含被动/兼容关节，消费者应读输出里的 **named layout**，勿假设 `qpos` 列数等于驱动关节数。

### 运行入口

```bash
# CLI（Kimodo）
core-retarget run examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz \
  --robot g1 --output runs/kimodo-g1 --video --thumbnail

# CLI（GEM-X，须 --fps）
core-retarget run examples/motions/gem-x/rapid_stepping.pt \
  --robot g1 --fps 30 --output runs/gemx-g1 --video --thumbnail

# 后端 / 本地网页
core-retarget backend --require-native
core-retarget serve   # 默认 http://127.0.0.1:8000
```

Python：`from core_retarget import Retargeter, RunConfig`。`backend="auto"` 优先编译 C++ MuJoCo 核，否则回退便携 Python。

### 许可分层（`docs/licenses.md`）

- 代码与 CoRe 场景包装：Apache-2.0
- 16 条捆绑示例动作：CC BY 4.0（Copyright 2026 Taemoon Jeong）；Kimodo 8 + GEM-X 8，不捆绑生成器权重
- 机器人 XML：Unitree BSD-3-Clause；K1 / Apollo / N1 / T1 / Oli Apache-2.0；ADAM Lite MIT；PM01 BSD-3-Clause（来自 GMR 仓修订）
- 研究软件：上真机前须在仿真中检查生成运动

## 对 wiki 的映射

- [`wiki/entities/core-retarget.md`](../../wiki/entities/core-retarget.md) — 软件实体
- [`wiki/entities/paper-core.md`](../../wiki/entities/paper-core.md) — Humanoids 2025 论文
- [`wiki/entities/paper-rmr.md`](../../wiki/entities/paper-rmr.md) — IROS 2025 论文（DMR / common-rigging）
- 交叉：[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[SOMA Retargeter](../../wiki/entities/soma-retargeter.md)、[Kimodo](../../wiki/entities/kimodo.md)
