---
type: entity
tags:
- paper
- humanoid
- motion-retargeting
- contact-aware
- reinforcement-learning
- sim2real
- korea-university
- kist
- uiuc
- repo
- mujoco
- soma
- kimodo
- unitree-g1
status: complete
updated: 2026-10-06
doi: 10.1109/Humanoids65713.2025.11203055
venue: Humanoids 2025
code: https://github.com/tmjeong1103/CoRe
related:
- ./paper-rmr.md
- ../concepts/motion-retargeting.md
- ../concepts/motion-retargeting-pipeline.md
- ../methods/motion-retargeting-gmr.md
- ../methods/reactor-physics-aware-motion-retargeting.md
- ./kimodo.md
- ./paper-physcore.md
- ./soma-retargeter.md
- ./robot-retargeter.md
- ./soma-x.md
- ./unitree-g1.md
- ./mujoco.md
sources:
- ../../sources/papers/core_humanoids_2025.md
- ../../sources/sites/core-page.md
- ../../sources/repos/core_retarget.md
- ../../sources/sites/huggingface-robotaemoon-core.md
- ../../sources/sites/rmr-page.md
summary: CoRe（Humanoids 2025，高丽大学/KIST/UIUC）：接触感知优化精炼 + 接触奖励 RL，先修脚滑/浮空再跟踪；软件 v0.1.0 已开源重定向与精炼，T2M 与 RL 训练未随仓发布。勿与 PhysCoRe 混淆。
project_id: core
---

# CoRe（接触感知优化与学习的人形运动）

**CoRe**（*Contact-aware motion Refinement*；论文 *CoRe: A Hybrid Approach of Contact-Aware Optimization and Learning for Humanoid Robot Motions*，[Humanoids 2025](https://doi.org/10.1109/Humanoids65713.2025.11203055)，[项目页](https://tmjeong1103.github.io/CoRe-page/)）由 **高丽大学（Korea University）**、**韩国科学技术研究院（KIST）**、**伊利诺伊大学厄巴纳-香槟分校（UIUC）** 提出：在 RL 跟踪之前，用接触段检测与接触约束优化把文本生成的人体运动修成可执行参考，再以接触感知奖励训策略。

> **同名消歧：** 本文是人形 **运动重定向 + 精炼 + RL**。可变形世界模型见 [PhysCoRe](./paper-physcore.md)。工程实现见 [CoRe 软件](#项目资源与工程补充)。

## 一句话定义

**先把参考修到接触可行，再让 RL 学跟踪——而不是把脚滑和浮空留给策略去补。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CoRe | Contact-aware motion Refinement | 本文接触感知精炼与混合管线 |
| RL | Reinforcement Learning | 精炼之后的物理模仿阶段 |
| T2M | Text-to-Motion | 管线最上游的文生人体运动 |
| IK | Inverse Kinematics | 机型重定向与落脚求解 |
| Sim2Real | Simulation to Real | 项目页展示的真机迁移 |

| DMR | Direction-based Motion Retargeting | 来自 [RMR](./paper-rmr.md) 的方向向量重定向阶段 |
| SOMA | Standardized Open Motion Avatar | NVIDIA 统一人体骨架；本工具吃 SOMA77 |
| FPA | Foot-Placement Adjustment | 接触感知落脚目标 + IK / 接地 |
| ARA | Absolute Root Adjustment | 根轨迹与接地偏置调整 |

## 为什么重要

- **打在「只靠 RL」的痛点：** 文生运动看起来像人，但初始运动学不可行会让跟踪不稳。CoRe 把脚滑、浮空、过加速当作 **参考层问题**。
- **精炼与学习分工清楚：** 优化管接触与碰撞，RL 管鲁棒执行；对应 [Pipeline](../concepts/motion-retargeting-pipeline.md) 的「几何映射 → 物理修补 → 跟踪」。
- **跨具身、少调参：** 项目页称同一管线覆盖全身 / 轮式 / 上身人形，无需逐任务调参或动力学级优化。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 高丽大学（Korea University）；韩国科学技术研究院（KIST）；伊利诺伊大学厄巴纳-香槟分校（UIUC） |
| **会议** | Humanoids 2025，pp. 293–300 |
| **平台** | 项目页：全身、轮式、上身三类人形；软件侧另绑 11 台商用人形 |
| **开源** | **部分开源：** [tmjeong1103/CoRe v0.1.0](https://github.com/tmjeong1103/CoRe/releases/tag/v0.1.0) 覆盖重定向+精炼；**T2M 与 RL 训练未发布** |
| **预印本** | 截至 2026-08-15 **无 arXiv**；以项目页 + IEEE 为准 |

## 核心原理 / 方法栈

```mermaid
flowchart TB
  t["自然语言"] --> t2m["文生人体运动"]
  t2m --> ret["机型重定向\nRMR / DMR"]
  ret --> det["接触段检测\n趾轨迹 C_f"]
  det --> opt["接触约束轨迹优化"]
  opt --> yaw["足偏航调整"]
  yaw --> col["自碰处理 + 平滑"]
  col --> rl["接触感知奖励 RL"]
  rl --> real["仿真 / 真机"]
```

1. **Contact Segment Detection** — 趾轨迹识别可靠足–地接触。
2. **Contact-Constrained Trajectory Optimization** — 消脚滑与浮空，平滑基座。
3. **Feet Orientation Adjustment** — 支撑相足偏航。
4. **Collision-handling and Smoothing** — 自碰位置修正与突变抑制。
5. **RL** — 精炼运动 + 接触段进入模仿学习，奖励显式对齐接触。

前端重定向的跨骨架统一见姊妹工作 [RMR](./paper-rmr.md)。

## 源码运行时序图

论文宣称的 T2M / RL **不在** 公开仓。可运行路径是软件仓的精炼管线，节点对齐 [`sources/repos/core_retarget.md`](../../sources/repos/core_retarget.md)：

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Src as Kimodo .npz / GEM-X .pt
    participant Run as core-retarget run
    participant DMR as stages/ DMR
    participant Ref as ARA / FPA / 自碰
    participant NPZ as final/robot_motion.npz
    Note over Dev,NPZ: 论文 T2M 与 RL 训练入口未发布
    Dev->>Src: 准备 SOMA 源运动
    Dev->>Run: --robot + --output
    Run->>DMR: SOMA77 身体目标
    DMR->>Ref: 机型 qpos 初值
    Ref->>NPZ: 接触精炼终档
    Note over Dev,NPZ: 下游跟踪需自接 WBT / AMP
```

- **不要期待：** 仓内没有论文级 text-to-motion 或 PPO 训练脚本。

## 工程实践

| 项 | 读法 |
|----|------|
| 何时用论文叙事 | 文生/估计人体运动要进 RL，且已观察到脚滑、浮空 |
| 何时用软件 | 已有 Kimodo / GEM-X SOMA 文件，要多机预览与安全 `.npz` |
| 与 GMR 分工 | GMR 覆盖格式与在线遥操；CoRe 覆盖 SOMA 输入 + 接触制品 |
| 真机 | 项目页展示 sim-to-real；软件仍标注「先仿真再上机」 |
| 复现数字 | IEEE 全文表格未开放；先用项目页视频与软件示例验收 |

## 实验与评测

项目页（非 IEEE 全文表）给出的证据：

- 同一管线迁移到 **全身、轮式、上身** 三类具身。
- 任务跨度：上身手势 → 全身 locomotion。
- 声称 **无任务特定调参、无动力学级优化**。
- 展示仿真到真机的可迁移性。

量化对比表待 IEEE / 预印本补录。

## 结论

**真正拉开差距的是「RL 之前把接触修对」，而不是再堆一个跟踪奖励；开源仓目前只兑现了精炼前半段。**

1. **真影响：参考层接触** — 脚滑 / 浮空应在优化里消，而不是交给策略硬补。
2. **真影响：接触进奖励** — 精炼段与 RL 奖励共用接触日程，避免「修了参考、训时又对不齐」。
3. **真影响：跨具身少调参** — 适合先看三类平台视频，再决定是否接入自有跟踪栈。
4. **次要代价：全文数字在付费墙后** — 选型先看视频与软件输出，不要引用未核对的 IEEE 表。
5. **部署读法：软件 ≠ 论文全栈** — v0.1.0 可出参考轨迹；T2M 与 RL 需自接 [Kimodo](./kimodo.md) / [BeyondMimic](../methods/beyondmimic.md) 等。
6. **工程读法：先跑 HF Space** — 用捆绑 `foot_walk_stop` / `scurry_walk` 看 11 机接触是否可接受。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [GMR](../methods/motion-retargeting-gmr.md) | 运动学前端；CoRe 在其后加接触优化，并主张再接 RL |
| [RMR](./paper-rmr.md) | 姊妹：canonical rig + 方向向量；CoRe 接接触精炼与学习 |
| [ReActor](../methods/reactor-physics-aware-motion-retargeting.md) | 参考形变与 RL **同环**；CoRe 是 **先优化、再 RL** |
| [DynaRetarget](../methods/dynaretarget-sbto-motion-retargeting.md) / [KDMR](./paper-kdmr.md) | 动力学 / GRF 级 TO；CoRe 自称不做动力学级优化 |
| [PhysCoRe](./paper-physcore.md) | 仅同名；对象是可变形世界模型 |

## 局限与风险

- **开源不完整：** 无法复现论文 RL 表；只能复现精炼软件。
- **无预印本：** 接触检测阈值、奖励权重等以 IEEE 为准，项目页只有定性步骤。
- **精炼仍是运动学+接触几何：** 不替代全身动力学可行化（对照 KDMR / DSMS / SBTO）。
- **输入依赖生成/估计质量：** 上游 T2M 漂得厉害时，接触段检测会跟着错。

## 项目资源与工程补充

### 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 高丽大学（Korea University）；联合韩国科学技术研究院（KIST）、伊利诺伊大学厄巴纳-香槟分校（UIUC） |
| **许可** | 代码 Apache-2.0；示例动作 CC BY 4.0；机器人 XML 保留厂商许可 |
| **平台** | macOS / Ubuntu，Python 3.10–3.13；MuJoCo 3.6.0 |
| **开源** | **已开源、可运行**（重定向 + 精炼 + 网页）；论文 **T2M / RL 训练未随 v0.1.0 发布** |
| **接口** | `Retargeter` Python API、`core-retarget` CLI、`core-retarget serve` / HF Space |

### 核心原理

扩展名即适配器：`.npz` → Kimodo（已评估全局 SOMA77）；`.pt` → GEM-X（body parameters + 接触 logits，固定 bind rig 求值后做 Z-up 与时变支撑地面归一）。两者进入同一不可变 SOMA77，再跑 DMR 与接触精炼。

```mermaid
flowchart LR
  k["Kimodo .npz"] --> s["SOMA77"]
  g["GEM-X .pt"] --> s
  s --> d["DMR\n方向向量重定向"]
  d --> c["接触精炼\nARA + FPA + 自碰"]
  c --> o["core-robot-motion-v1\n.npz + 预览"]
```

#### 九段制品

| 阶段 | 制品 | 职责 |
|------|------|------|
| 1 | `1_contacts.npz` | 源校验 + 提供方感知足接触 |
| 2 | `2_dmr.npz` | 身体目标 → 选定机器人 |
| 3 | `3_initial_collision.npz` | 初始手臂自碰 |
| 4–5 | 轨迹 / `5_ara.npz` | 根、踝、足底、趾；根与接地偏置 |
| 6–7 | FPA | 接触感知落脚目标 + IK |
| 8–9 | 终态 | 再一次手臂自碰 + 诊断后写出终档 |

输出含时间戳、MuJoCo `qpos`、命名根/关节布局、接触与源/模型哈希；**无 object 数组**。部分厂商模型 `nq` ≠ 驱动 DoF，必须读 named layout。

### 源码运行时序图

官方入口对齐仓库 `core_retarget/` 与 README：`core-retarget run` / `Retargeter.run` / `core-retarget serve` 都调用同一 `run_retarget_pipeline()`。

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者 / Space 用户
    participant CLI as core-retarget run / serve
    participant Adp as motion/ 适配器
    participant Stg as stages/ 九段
    participant MJ as mujoco/ 核<br/>native 或 python
    participant Exp as export/<br/>core-robot-motion-v1
    participant Out as runs/.../final
    Dev->>CLI: Kimodo .npz 或 GEM-X .pt + --robot
    CLI->>Adp: load_source_motion / validate
    Adp-->>Stg: 不可变 SOMA77 + 接触日程
    Stg->>MJ: DMR / 碰撞距离 / 足端 IK
    MJ-->>Stg: qpos 与碰撞诊断
    Stg->>Exp: 9_diagnostics 后写终档
    Exp->>Out: robot_motion.npz + manifest
    opt 预览
      CLI->>MJ: 无头渲染 MP4 / PNG
    end
```

- **最短复现：** `pip install -e ".[gemx,video]"` → `core-retarget backend --require-native` → `core-retarget run examples/motions/kimodo/... --robot g1 --video`。
- **浏览器：** `pip install -e ".[web]"` → `core-retarget serve`，或打开 HF Space。
- **批量：** `scripts/generate_example_outputs.py --source-set kimodo|gem-x`。

### 工程实践

| 项 | 建议 |
|----|------|
| 源格式 | Kimodo 用嵌入或默认 FPS；GEM-X **必须** `--fps`（捆绑示例 30 Hz） |
| 后端 | `auto` 优先 C++ 核；清单记录 requested / selected backend |
| 安全加载 | `.npz` 关 pickle；`.pt` 用 `weights_only=True`；输出再 `allow_pickle=False` 校验 |
| 资产 | `core-retarget robots verify` 核 XML/网格哈希；勿改 vendor 目录 |
| 真机 | README 写明 **研究软件**：先仿真检查再上硬件 |
| 许可 | 再分发时分开代码、示例动作与各厂商 `SOURCE.yaml` |

#### 捆绑机型（v0.1.0）

`g1` / `h1` / `h2` / `r1`（Unitree）、`k1`（ROBOTIS）、`apollo`（Apptronik）、`oli`（LimX）、`n1`（Fourier）、`adam`（PNDbotics）、`t1`（Booster）、`pm01`（ENGINEAI）。

### 局限与风险

- **不是论文全管线：** v0.1.0 **不含** text-to-motion 与 contact-aware RL；产物是运动学+接触精炼参考，下游仍要 [WBT](../concepts/whole-body-tracking-pipeline.md) / AMP。
- **输入契约窄：** 只吃 SOMA77 Kimodo/GEM-X；SMPL-X / BVH 需先经 [Kimodo](./kimodo.md) / [SOMA-X](./soma-x.md) / [SOMA Retargeter](./soma-retargeter.md) 转换。
- **接触质量不保证任意动作：** 回归基线钉在捆绑 Kimodo 参考上，不外推到任意 GEM-X 或自采视频。
- **安装要 C++17：** native 核在安装期编译；纯 Python 后端可跑但更慢。
- **厂商模型有本地修改：** 以各目录 `SOURCE.yaml` / `MODIFICATIONS.md` 为准，勿当官方仿真模型的 bit-exact 副本。

### 与相近工具对比

| 维度 | CoRe v0.1.0 | [GMR](../methods/motion-retargeting-gmr.md) | [SOMA Retargeter](./soma-retargeter.md) | [robot_retargeter](./robot-retargeter.md) |
|------|-------------|---------------------------------------------|----------------------------------------|------------------------------------------|
| 典型输入 | Kimodo `.npz` / GEM-X `.pt` | BVH / SMPL / FBX | SOMA BVH | SMPL-X `.npz` / LAFAN1 CSV |
| 接触精炼 | 九段 ARA/FPA/自碰 | 几何 IK，物理另补 | 足部稳定 + 限位 | 接触锁定 FrameTask |
| 多机 | 11 台捆绑 | 多机、格式广 | 主推 G1 | G1/H2/T800 等并排 |
| 浏览器演示 | HF Space / `serve` | 无官方 Space | 无 | 无 |

## 关联页面

- [RMR](./paper-rmr.md)
- [Motion Retargeting](../concepts/motion-retargeting.md) / [Pipeline](../concepts/motion-retargeting-pipeline.md)
- [GMR](../methods/motion-retargeting-gmr.md) / [ReActor](../methods/reactor-physics-aware-motion-retargeting.md)
- [Kimodo](./kimodo.md)
- [PhysCoRe（同名消歧）](./paper-physcore.md)

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)
- [GMR](../methods/motion-retargeting-gmr.md)
- [SOMA Retargeter](./soma-retargeter.md) / [robot_retargeter](./robot-retargeter.md)
- [Kimodo](./kimodo.md) / [SOMA-X](./soma-x.md) / [Unitree G1](./unitree-g1.md)

- [mujoco](./mujoco.md)

## 参考来源

- [CoRe Humanoids 2025 论文归档](../../sources/papers/core_humanoids_2025.md)
- [CoRe 项目页归档](../../sources/sites/core-page.md)
- [CoRe 仓库归档](../../sources/repos/core_retarget.md)

- [Hugging Face Space 归档](../../sources/sites/huggingface-robotaemoon-core.md)
- [RMR 项目页归档](../../sources/sites/rmr-page.md)

## 推荐继续阅读

- 项目页：<https://tmjeong1103.github.io/CoRe-page/>
- IEEE：<https://doi.org/10.1109/Humanoids65713.2025.11203055>
- 软件：<https://github.com/tmjeong1103/CoRe>

- 发布说明：<https://github.com/tmjeong1103/CoRe/releases/tag/v0.1.0>
- 在线演示：<https://huggingface.co/spaces/robotaemoon/CoRe>
- 架构文档：<https://github.com/tmjeong1103/CoRe/blob/main/docs/architecture.md>
