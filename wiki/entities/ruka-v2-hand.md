---

type: entity
tags: [manipulation, dexterous-hand, hardware, open-source, tendon-driven, teleoperation, imitation-learning, nyu]
status: complete
updated: 2026-06-13
related:
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
  - ../queries/dexterous-data-collection-guide.md
  - ./paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md
  - ./orca-hand.md
  - ./allegro-hand.md
sources:
  - ../../sources/papers/ruka_v2_arxiv_2603_26660.md
  - ../../sources/repos/ruka-v2.md
  - ../../sources/sites/ruka-hand-v2-github-io.md
summary: "RUKA-v2（NYU）：全栈开源腱驱动仿人灵巧手，16 主动指 DoF + 2-DoF 平行腕 + 指根外展/内收，材料约 1.5K USD；AnyTeleop 重定向 + OpenTeach VR 遥操作 + BAKU 模仿学习已验证。"
---

# RUKA-v2 Hand

## 一句话定义

**RUKA-v2** 是纽约大学团队发布的 **全硬件、全软件、全文档开源** 腱驱动仿人灵巧手：在 [RUKA v1](./paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md) 基础上增加 **解耦 2-DoF 平行腕** 与 **MCP 外展/内收**，材料成本约 **1,500 USD**；官方入口为项目页 [ruka-hand-v2.github.io](https://ruka-hand-v2.github.io/) 与代码库 [ruka-hand-v2/RUKA-v2](https://github.com/ruka-hand-v2/RUKA-v2)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DOF | Degrees of Freedom | 独立可控运动轴数量 |
| MCP | Metacarpophalangeal Joint | 掌指关节，指根屈伸与外展/内收常发生于此 |
| DIP / PIP | Distal / Proximal Interphalangeal Joint | 远侧 / 近侧指间关节 |
| BC | Behavior Cloning | 行为克隆，从专家示范监督学习策略 |
| URDF | Unified Robot Description Format | 统一机器人描述格式，用于仿真与重定向 |
| VR | Virtual Reality | 虚拟现实，论文遥操作使用 Oculus 头显 |

## 为什么重要

- **可及性：** 相对 Allegro（约 16K USD）、Sharpa（约 50K USD）等，RUKA-v2 以 **3D 打印 + 现货件** 把 **16 主动 DoF + 腕** 压到 **约 1.5K USD** 量级，且论文 Table 1 标注 **硬件与软件均全公开**（✓），适合高校 lab 复刻与改 CAD。
- **补齐 v1 短板：** v1 缺 **腕** 与 **指间外收**；用户研究显示 v2 遥操作 **完成时间 −51.3%**、**成功率 +21.2%**（相对 v1，三项任务）。
- **学习闭环已跑通：** 单臂 10 + 双臂 3 遥操作任务，以及 **BAKU** 视觉 BC 三任务（面包 pick-place、开音乐盒、捡笔），证明其作为 **数据采集 + IL 平台** 的可行性。
- **控制可复现：** 将 retargeting 与 joint→motor 映射 **解耦**；采用 **AnyTeleop 向量重定向** + **线性 motor 映射** + **自动 per-motor 校准**，并开源 **磁编码器** 套件以降低对动捕手套的依赖。

## 硬件要点

| 维度 | 规格（论文/项目页） |
|------|---------------------|
| 驱动 | **腱驱动（T）**；电机置于前臂 |
| 主动 DoF | **16**（指/拇指）+ **2**（腕） |
| 腕 | **平行 2-DoF**：屈伸 + 桡/尺偏；球铰共 pivot，腱过 rotation center |
| 外展 | 四指 MCP **adduction/abduction**；**中指固定**作参考 |
| 结构 | 主体 **3D 打印**；轴承/紧固件/弹簧现货 |
| 指尖 | **E-flesh** 软接触（3D 打印）；可选触觉 |
| 传感（可选） | **AS5600 磁编码器** 可拆卸套件（校准与 ground truth） |
| 安装 | **侧装** 腕法兰，便于桌面 7-DoF 臂 |
| 成本 | 材料 **约 1,500 USD**（论文）；项目页写 **低于 2,000 USD** |
| 耐久 | 论文：**>5 h** 连续运行无热限；静态载荷见下表 |

**静态载荷（摘要，15–20 s 保持）：** 非拇指 DIP–PIP **1200 g**；MCP **780 g**；外展 **150 g**；拇指 **835 g**；腕 supination/pronation **1215 g**。

## 软件与学习管线

```mermaid
flowchart LR
  A[人演示<br/>视频 / VR 手姿] --> B[AnyTeleop<br/>向量重定向]
  B --> C[RUKA-v2<br/>关节角 θ]
  C --> D[线性映射 +<br/>per-motor 校准]
  D --> E[腱驱动电机指令]
  E --> F[Franka 7-DoF + RUKA-v2<br/>OpenTeach 遥操作]
  F --> G[示范数据集<br/>~100 条/任务]
  G --> H[BAKU 视觉 BC<br/>RGB + proprio]
  H --> I[自主 rollout]
```

- **遥操作：** **OpenTeach** + **Oculus VR**；Franka 臂末端轨迹由 OpenTeach IK，手部经上述控制器。
- **模仿学习：** **BAKU**；观测 **23D proprio**（7 臂 + 16 手）+ 固定相机 **RGB**；收集时对关节加噪声以包含 **恢复行为**。
- **安装：** `git clone --recurse-submodules` → Conda `ruka_hand` → `pip install -e .`（详见 [sources/repos/ruka-v2.md](../../sources/repos/ruka-v2.md)）。

## 与相近平台对照（论文 Table 1 归纳）

| 平台 | 开源 | 成本量级 | 主动 DoF | 腕 DoF | 驱动 |
|------|------|----------|----------|--------|------|
| Allegro V4 | 部分 | 约 16K USD | 16 | 0 | 直驱 |
| RUKA v1 | ✓ | 约 1.3K USD | 11 | 0 | 腱 |
| ORCA | ✓ | 约 3.5K USD | 17 | 1 | 腱 |
| RUKA-v2 | ✓ | 约 1.5K USD | 16 | 2 | 腱 |
| Wuji Hand | ✗ | 约 5.5K USD | 20 | 0 | 直驱 |
| Sharpa Wave | ✗ | 约 50K USD | 22 | 0 | 直驱 |

## 常见误区

- **「v2 只是 v1 小改」** — v2 新增 **整腕模块 + 四指外展腱路**，DoF、腱路由与控制栈均变；仓库与 BOM **独立**（`ruka-hand-v2` vs `ruka-hand`）。
- **「必须买动捕手套才能用」** — v1 偏数据驱动 tendon 标定；v2 **默认 AnyTeleop 视觉重定向 + 线性 motor 映射**，并开源 **磁编码器** 做校准与评测。
- **「开源 = 只有 STL」** — 团队公开 **可编辑 CAD**、装配视频、控制器与校准脚本；工程上仍需要 **3D 打印与腱张紧** 经验（与多数腱驱动手相同）。

## 关联页面

- [Manipulation](../tasks/manipulation.md) — 灵巧操作任务语境
- [Teleoperation](../tasks/teleoperation.md) — VR 遥操作采集
- [灵巧操作数据采集指南](../queries/dexterous-data-collection-guide.md) — 开源手 + 视觉重定向选型
- [RUKA v1（Paper Notebooks 待深读）](./paper-notebook-ruka-rethinking-the-design-of-humanoid-hands-wit.md)
- [Orca Hand](./orca-hand.md) — 另一类开源腱驱动仿手
- [Allegro Hand](./allegro-hand.md) — 直驱科研平台对照

## 推荐继续阅读

- 项目页演示与 BibTeX：<https://ruka-hand-v2.github.io/>
- 论文 HTML：<https://arxiv.org/html/2603.26660>
- 代码与安装：<https://github.com/ruka-hand-v2/RUKA-v2>
- 前代 RUKA v1：<https://github.com/ruka-hand/RUKA> / <https://ruka-hand.github.io/>

## 参考来源

- [ruka_v2_arxiv_2603_26660.md](../../sources/papers/ruka_v2_arxiv_2603_26660.md)
- [ruka-v2.md](../../sources/repos/ruka-v2.md)
- [ruka-hand-v2-github-io.md](../../sources/sites/ruka-hand-v2-github-io.md)
