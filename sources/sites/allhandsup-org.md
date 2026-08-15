# All Hands Up !（allhandsup.org）

- **标题：** All Hands Up ! — RLWRLD Robot Hand Archive
- **类型：** site / hardware-archive
- **URL：** <https://allhandsup.org/zh/#gallery>（中文画廊；规范域 [allhandsup.rlwrld.co](https://allhandsup.rlwrld.co/en/)）
- **镜像 / 语言：** `/en/` · `/ko/` · `/ja/` · `/zh/`
- **机构：** 瑞沃世界（RLWRLD）
- **作者（页脚）：** Max Chang Hwan Kim · Mark Seungjae Lee
- **联系：** partnership@rlwrld.ai
- **入库日期：** 2026-08-15
- **配套产品：** [RLDX-1](https://rlwrld.ai/rldx-1)（同机构灵巧操作 VLA）；任务基准 [DexBench](https://dexbench.org)
- **代码：** 截至入库日 **无独立 GitHub 仓**（`RLWRLD/allhandsup`、`RLWRLD/AllHandsUp` 均 404）。画廊 URDF / 网格与 Kapandji JSON **可经站点 HTTP 下载**，许可按手标注（厂商 robot description 许可证，非站点统一许可证）

## 一句话摘要

RLWRLD 基于「真机操作十数款腕装模块化灵巧手」的经验，公开的 **浏览器内 URDF 画廊 + 规格对照 + 仿真 Kapandji 对掌评分 + 硬件设计长文**；用来补规格表读不出的任务级取舍，而不是再发一份 DoF/握力清单。

## 开源状态（步骤 2.5，截至 2026-08-15）

| 项 | 结论 |
|----|------|
| 项目页 | **已开放**：[allhandsup.org](https://allhandsup.org/zh/#gallery) / [allhandsup.rlwrld.co](https://allhandsup.rlwrld.co/en/) |
| 独立代码仓 | **未开源**：站点页脚与 HTML 未列 GitHub；检索官方组织仓名为 404 |
| URDF / 网格 | **部分开放**：`hands_urdf/_registry.json` + 各手目录可 HTTP 拉取；许可写在 registry 的 `license` 字段（MIT / Apache-2.0 / BSD / 木兰等，按厂商） |
| Kapandji 数据 | **已公开**：`hands_urdf/_kapandji.json`（`tools/build_kapandji_json.py` 生成；仿真扫掠，非临床实测） |
| DexBench 任务分 | **不在本站完整发布**：长文引用 [dexbench.org](https://dexbench.org) 的 18 项任务（T00–T17）；本站只给硬件轴与少数产品叙事 |
| 长文 Part 3 | **Coming Soon** |

**判定：部分开源 / 公开可访问。** 可视化与描述文件可复用，但不是带训练/驱动入口的可运行软件栈；勿写成「官方 GitHub 已开源」。

## 为何值得保留

- **规格表缺口**：拇指 Kapandji 可达、DIP 是否独立驱动、最小可抓直径、指垫摩擦/硬度，决定任务成败，却很少印在 datasheet。
- **腕装模块手对照**：明确排除前臂集成腱驱（如 ALLEX）作为主比较集，避免和 7 轴通用臂「拧上手」的安装约束混谈。
- **双类型选型**：承认尚无完美手，拆成 Type 1（部署：轻、耐用、中等背驱）与 Type 2（采数：更高背驱、力矩可当学习信号）。
- **与本库已有手实体对齐**：画廊含 Allegro V5 Plus、Orca V1、Wuji Hand V1.1 等，可挂到已有硬件页，而不是另造产品百科。

## 公开要点（编译自画廊 registry + 长文，截至入库日）

### 站点结构

| 入口 | 内容 |
|------|------|
| `#gallery` | 缩略图网格 → 详情：Three.js URDF 关节滑条、Open/Spread/Grasp/V 预设、Kapandji K0–K10 姿态按钮 |
| §1 *Why Dexterity Matters* | 三轴（形态多样性 / 握力 / 力觉与背驱）及尺寸–出力–背驱耦合；三则 ALLEX URDF 示意仿真（非该手实测能力） |
| §2 *What Makes a Good Robot Hand* | 评测项：拇指 ROM、DIP 独立、最小直径、指垫材料；SharpaWave / DG-5F-S / Wuji V1.1 叙事 + Type 1/2 规格靶标 |
| §3 | Coming Soon |
| 页脚 | 欢迎厂商寄样测评；声明画廊手图为 **AI 生成**，可能与实物不完全一致 |

### 画廊 16 手（`hands_urdf/_registry.json`）

规格列来自 registry `specs`（厂商/站点汇编，单位：重量 g、钩挂载荷 kg、指尖力 N）；**Kapandji** 为仿真达到的里程碑数（满分 11 = K0–K10）。`DoF` 为 specs 宣称主动自由度，与 URDF `movableCount` 不必相等（欠驱动 / mimic 会拉开）。

| 画廊 id | 显示名 | 厂商 | DoF | 重量 | 背驱 | Kapandji | 描述许可（摘） |
|---------|--------|------|-----|------|------|----------|----------------|
| Agibot-Omnihand-Pro | Agibot Omnihand Pro | Agibot | 12 | 750 | No | 10 | 木兰宽松 v2 |
| Allegro-V5-Plus | Allegro V5 Plus | Wonik Robotics | 16 | 1024 | Yes | 9 | BSD-2-Clause |
| Brainco-Revo2 | Brainco Revo2 | BrainCo | 11（6 主动） | 383 | — | 5 | Apache-2.0 |
| Brainco-Revo3 | Brainco Revo3 | BrainCo | 21 | — | Yes | 11 | 厂商提供（页未写 SPDX） |
| DG-5F-M | DG-5F-M | Tesollo | 20 | 1763 | Yes | 10 | Tesollo 提供 |
| DG-5F-S | DG-5F-S | Tesollo | 20 | 880 | Yes | 10 | Tesollo 提供 |
| Inspire-F1 | Inspire RH56F1 | Inspire Robotics | 6 | 630 | Yes | 5 | MIT |
| Leap-Hand-V1 | LEAP Hand V1 | Carnegie Mellon | 16 | 595 | Yes | 9 | MIT |
| Linkerbot-L20 | Linkerbot L20 | Linkerbot | 16 | 1000 | No | 8 | Apache-2.0 |
| OYMotion-ROHand | OYMotion ROHand | OYMotion | 6 主动 | 680 | — | 6 | MIT |
| Orca-V1 | Orca Hand V1 | ETH Zurich SRL | 16 | 1100 | Yes | 9 | MIT |
| Psyonic-Ability-Hand | Psyonic Ability Hand | PSYONIC | 6 | 490 | — | 5 | MIT |
| Robotis-HX5-D20 | Robotis HX5-D20 | ROBOTIS | 20 | 1360 | Yes | 11 | Apache-2.0 |
| SharpaWave | Sharpa Wave | Sharpa Robotics | 22 | 1300 | Yes | 9 | Apache-2.0 |
| Wuji-V1.1 | Wuji Hand V1.1 | Wuji Tech | 20 | 590 | No | 10 | MIT |
| xHand-1 | XHAND 1 | Robotera | 12 | 1100 | Yes | 8 | BSD-3-Clause |

**Kapandji 方法（站点原文）：** 在该手自己的 URDF 上扫掠拇指，接触到 11 个临床对掌地标即计 1 分；按钮姿态取「最干净」的接触位。`_kapandji.json` 的 `_validation` 记录部分姿态 **超出 URDF 关节限位**（DG-5F-M、LEAP、Orca、Wuji），读分时需打折。

### 长文评测叙事（§2，非画廊全表）

- **SharpaWave**：22 DoF、背驱、手长 200 mm、约 1200 g、指尖力 20 N；小指 CMC 主动、拇指 CMC+MCP 独立；指尖 >1000 触觉像素；DexBench 力相关任务表现高；单价约 **$50,000**。
- **DG-5F-S**：20 DoF、背驱、208.5 mm、880 g、19.5 N；约 **$7,500**；DexBench 与 Sharpa 接近；指截面矩形、金属壳体摩擦系数低，美工刀类工具不稳。
- **Wuji Hand V1.1**：20 DoF、**不背驱**、195 mm、590 g、15 N；接近人手尺度；T16 等小工具（4 mm 内六角）表现好；意外冲击会伤关节，采数连续性风险。
- **ALLEX（旁注，前臂集成，不入画廊主集）**：15 DoF、背驱、指尖力 40 N；约 100 g 接触力可从关节力矩估计；亦称可抓 4 mm 内六角。

### Type 1 / Type 2 靶标（§2 表，内部选型语言）

两端都要求主动 DoF ≥20、四指/拇指各 ≥4、钩挂最大载荷 ≥15 kg（30 s）/ 连续 ≥5 kg、闭合 ≤0.3 s、重复性 ≤±0.2 mm、全掌触觉与 ≥100 Hz 位置环等。差别主要在：

| 属性 | Type 1（偏部署） | Type 2（偏采数） |
|------|------------------|------------------|
| 外形 | 更短更窄更轻（长 <205 mm、重 <600 g） | 略放宽（长 <220 mm、重 <900 g） |
| 背驱力矩上限 | 更高（MCP ≤0.4–0.5 N·m） | 更低阻抗（MCP ≤0.15–0.25 N·m） |
| 指尖力 | ≥15 N | ≥12 N |

新闻稿（2026-07）称平台会 **按季度更新** 实证数据。

## 关联资料

- 同机构 VLA：[RLDX-1 仓库归档](../repos/rldx-1.md)
- 任务基准（独立站，本次不另建页）：<https://dexbench.org>

## 对 wiki 的映射

- [All Hands Up](../../wiki/entities/all-hands-up.md) — 平台实体与选型读法
- 交叉：[Allegro Hand](../../wiki/entities/allegro-hand.md)、[Orca Hand](../../wiki/entities/orca-hand.md)、[舞肌 / Wuji Hand](../../wiki/entities/wuji-robotics.md)、[RLDX-1](../../wiki/entities/rldx-1.md)、[灵巧手运动学](../../wiki/concepts/dexterous-kinematics.md)、[灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)
